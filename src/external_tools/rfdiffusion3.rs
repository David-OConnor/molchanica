//! [RFdiffusion3](https://github.com/RosettaCommons/foundry) backbone generation.
//!
//! RFD3 generates backbone coordinates around whatever context it is given — nothing at all, a
//! target to bind, a ligand to build a pocket for, a motif to scaffold, a nucleic acid, a point
//! group to obey. Its output is coordinates without a sequence, which is exactly what the MPNN
//! family in [`super::mpnn`] takes as input, so the two together are a design pipeline that starts
//! and ends inside Molchanica.
//!
//! # Why this looks different to the other adapters
//!
//! RFdiffusion 1 and 2 were driven entirely by Hydra overrides on the command line. RFD3 is not:
//! the design itself is a JSON `InputSpecification` — one object per named design, holding the
//! contig, the ligands, and a per-atom account of what is held fixed — and only the job-level
//! settings stay on the command line. Molchanica supports RFD3 alone; there is no migration path
//! from the older command lines, and `bio_tools` no longer installs them.
//!
//! The set of inputs, their labels, bounds, defaults, and help text are not restated here: they
//! come from `bio_tools`' shared form contract through [`super::tool_form`], the same one
//! `bio_web` renders. What is here is the part that is genuinely RFD3's own — turning those form
//! values into an `InputSpecification` and a command line, with the same validation the hosted
//! form applies, so a mistake is reported in the window rather than several minutes into a run.
//!
//! Reference: <https://rosettacommons.github.io/foundry/models/rfd3/input.html>

use std::{
    collections::HashMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use bio_tools::tool_definitions::presets;
use regex::Regex;
use serde_json::{Map, Value};

use crate::external_tools::{
    Tool, bundle_root, data_root, find_executable, run_to_completion_logged,
    tool_form::FormContract,
};

/// The name foundry's checkpoint registry gives the RFD3 weights, and so the name `bio_tools`
/// downloads them under, inside the tool's bundle directory.
const CHECKPOINT: &str = "checkpoints/rfd3_latest.ckpt";

/// The prefix a preset uses to name a structure `bio_tools` bundles with it.
const BUNDLED_PREFIX: &str = "bio-tools://rfd3/";

/// The value an `input` takes to mean "the structure supplied alongside the specification", which
/// for Molchanica is the peptide picked in the window rather than a web upload.
const SUPPLIED_INPUT: &str = "uploaded";

/// A structure to design against that is not already a file on disk — the peptide selected in the
/// window, rendered to PDB.
#[derive(Clone, Debug)]
pub struct SuppliedStructure {
    /// Used to name the file written into the run directory.
    pub name: String,
    pub pdb: String,
}

/// How the values in a request are to be read. Mirrors `bio_tools`' `input_mode` selector.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InputMode {
    /// Build one `InputSpecification` entry out of the individual parameter fields.
    #[default]
    Parameters,
    /// Take a complete JSON (or YAML) document of named designs as written.
    Document,
}

impl InputMode {
    /// The `input_mode` value `bio_tools`' contract uses for this mode.
    pub fn contract_value(self) -> &'static str {
        match self {
            Self::Parameters => "parameters",
            Self::Document => "text",
        }
    }

    pub fn from_contract_value(value: &str) -> Self {
        match value {
            // The hosted form separates pasting a document from uploading one; on the desktop both
            // end up as the same text, because opening the file is a file picker rather than a
            // transfer, so `upload` folds into `Document`.
            "text" | "upload" => Self::Document,
            _ => Self::Parameters,
        }
    }
}

/// One RFD3 run, as the window has it configured.
#[derive(Clone, Debug, Default)]
pub struct Rfd3Request {
    pub input_mode: InputMode,
    /// Form values keyed by the field names in `bio_tools`' `rfd3` contract.
    pub values: HashMap<String, String>,
    /// What an `input` of `"uploaded"` refers to.
    pub supplied_structure: Option<SuppliedStructure>,
}

/// What a finished run left behind.
#[derive(Clone, Debug)]
pub struct Rfd3Result {
    /// The directory holding the inputs, the outputs, and the log. Kept after the run: designs are
    /// the point of it, and they are expensive to reproduce.
    pub run_dir: PathBuf,
    /// Generated backbones, in the order they were found.
    pub structures: Vec<PathBuf>,
    /// The `InputSpecification` actually submitted, for the window to show.
    pub specification: String,
    /// The command line, rendered as it was run.
    pub command: String,
}

/// Run RFdiffusion3 to completion.
///
/// Blocking, and slow enough — minutes to hours — that it must be called from a worker thread.
pub fn run(request: &Rfd3Request) -> io::Result<Rfd3Result> {
    let executable = find_executable(Tool::RfDiffusion3)?;
    let checkpoint = bundle_root(Tool::RfDiffusion3)?.join(CHECKPOINT);
    if !checkpoint.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "the RFdiffusion3 checkpoint is missing from {}. Re-install RFdiffusion3 from \
                 Molchanica's Tools panel.",
                checkpoint.display()
            ),
        ));
    }

    let form = Form::new(&request.values);
    let job = job_name(&form);
    let run_dir = create_run_dir(&job)?;

    let plan = build(request, &form, &job, &run_dir).map_err(invalid_input)?;
    let inputs_path = run_dir.join(plan.inputs_file_name);
    fs::write(&inputs_path, &plan.inputs_document)?;

    let output = run_dir.join("output");
    let mut command = Command::new(&executable);
    command
        .arg("design")
        .arg(format!("out_dir={}", output.display()))
        .arg(format!("inputs={}", inputs_path.display()))
        .arg(format!("ckpt_path={}", checkpoint.display()));
    for (key, value) in &plan.overrides {
        command.arg(format!("{key}={value}"));
    }
    command.current_dir(&run_dir);
    let rendered = crate::external_tools::display_command(&command);

    let log = run_to_completion_logged(&mut command, "RFdiffusion3", "backbone generation")?;
    let _ = fs::write(run_dir.join("rfd3.log"), &log);

    Ok(Rfd3Result {
        structures: generated_structures(&run_dir),
        specification: plan.inputs_document,
        command: rendered,
        run_dir,
    })
}

/// Everything a run needs that is derived from the form, so that it can be built — and its errors
/// reported — without starting a process.
struct RunPlan {
    inputs_file_name: &'static str,
    inputs_document: String,
    overrides: Vec<(String, String)>,
}

fn build(
    request: &Rfd3Request,
    form: &Form<'_>,
    job: &str,
    run_dir: &Path,
) -> Result<RunPlan, String> {
    let overrides_object = json_object(form.text("specification"), "specification")?;

    let (inputs_file_name, inputs_document, specification) = match request.input_mode {
        InputMode::Parameters => {
            let mut entry = native_specification(form)?;
            for (key, value) in &overrides_object {
                entry.insert(key.clone(), value.clone());
            }
            let mut document = Map::new();
            document.insert(job.to_owned(), Value::Object(entry));
            let document = resolve_inputs(document, request, run_dir)?;
            let text = serde_json::to_string_pretty(&Value::Object(document.clone()))
                .map_err(|error| error.to_string())?;
            ("inputs.json", text, Some(document))
        }
        InputMode::Document => {
            let raw = form.raw("inputs");
            if raw.trim().is_empty() {
                return Err(
                    "Enter an input document, or switch to \"Set parameters here\".".into(),
                );
            }
            match serde_json::from_str::<Value>(raw) {
                Ok(Value::Object(mut document)) => {
                    if document.is_empty() {
                        return Err("The input document has no named designs in it.".into());
                    }
                    for entry in document.values_mut() {
                        let Value::Object(entry) = entry else {
                            return Err(
                                "Every entry in the input document must be an object.".into()
                            );
                        };
                        for (key, value) in &overrides_object {
                            entry.insert(key.clone(), value.clone());
                        }
                    }
                    let document = resolve_inputs(document, request, run_dir)?;
                    let text = serde_json::to_string_pretty(&Value::Object(document.clone()))
                        .map_err(|error| error.to_string())?;
                    ("inputs.json", text, Some(document))
                }
                Ok(_) => {
                    return Err("The input document must be an object of named designs.".into());
                }
                // Not JSON. RFD3 reads YAML too, so rather than rejecting a document it would have
                // accepted, hand it over as written. Nothing in it can be rewritten in that case,
                // so `input` has to name a path on this machine.
                Err(_) => {
                    if !overrides_object.is_empty() {
                        return Err("Specification overrides need a JSON input document; \
                                    they cannot be merged into YAML."
                            .into());
                    }
                    ("inputs.yaml", raw.to_owned(), None)
                }
            }
        }
    };

    let subset = string_list(form.raw("json_keys_subset"), "json_keys_subset")?;
    if let Some(document) = &specification {
        for name in &subset {
            if !document.contains_key(name) {
                return Err(format!(
                    "json_keys_subset names \"{name}\", which is not a design in the input document."
                ));
            }
        }
    }
    let symmetric = match &specification {
        Some(document) => symmetry_check(document, &subset, form)?,
        None => false,
    };

    let mut overrides = vec![
        (
            "n_batches".to_owned(),
            form.int("n_batches", 1, 1, 100)?.to_string(),
        ),
        (
            "diffusion_batch_size".to_owned(),
            form.int("diffusion_batch_size", 8, 1, 64)?.to_string(),
        ),
        (
            "inference_sampler.num_timesteps".to_owned(),
            form.int("inference_sampler.num_timesteps", 200, 20, 1_000)?
                .to_string(),
        ),
        (
            "inference_sampler.step_scale".to_owned(),
            encode(&number(form.float(
                "inference_sampler.step_scale",
                1.5,
                0.1,
                10.0,
            )?)),
        ),
        (
            "inference_sampler.gamma_0".to_owned(),
            encode(&number(form.float(
                "inference_sampler.gamma_0",
                0.6,
                0.0,
                1.0,
            )?)),
        ),
        (
            "low_memory_mode".to_owned(),
            encode(&Value::Bool(form.boolean("low_memory_mode", false)?)),
        ),
        (
            "dump_trajectories".to_owned(),
            encode(&Value::Bool(form.boolean("dump_trajectories", false)?)),
        ),
        (
            "json_keys_subset".to_owned(),
            encode(&if subset.is_empty() {
                Value::Null
            } else {
                Value::Array(
                    subset
                        .iter()
                        .map(|name| Value::String(name.clone()))
                        .collect(),
                )
            }),
        ),
    ];
    overrides.extend(sampler_options(form, symmetric)?);

    for (key, default) in [
        ("prevalidate_inputs", false),
        ("cleanup_guideposts", true),
        ("cleanup_virtual_atoms", true),
        ("read_sequence_from_sequence_head", true),
        ("output_full_json", true),
        ("dump_prediction_metadata_json", true),
        ("align_trajectory_structures", false),
    ] {
        overrides.push((
            key.to_owned(),
            encode(&Value::Bool(form.boolean(key, default)?)),
        ));
    }

    let prefix = form.text("global_prefix");
    if !prefix.is_empty() {
        // `""` is how the upstream CLI is told to use no prefix at all, as distinct from leaving
        // the argument off and taking the model's default.
        let prefix = if prefix == "\"\"" { "" } else { prefix };
        if !prefix.is_empty() && !is_plain_name(prefix) {
            return Err(
                "global_prefix must use letters, digits, dots, underscores or hyphens.".into(),
            );
        }
        overrides.push((
            "global_prefix".to_owned(),
            encode(&Value::String(prefix.to_owned())),
        ));
    }

    Ok(RunPlan {
        inputs_file_name,
        inputs_document,
        overrides,
    })
}

// ---------------------------------------------------------------------------------------------
// The InputSpecification
// ---------------------------------------------------------------------------------------------

/// Build one `InputSpecification` entry from the individual parameter fields.
///
/// Follows the documented field contract rather than any workflow-specific shorthand:
/// <https://rosettacommons.github.io/foundry/models/rfd3/input.html#inputspecification-fields>
fn native_specification(form: &Form<'_>) -> Result<Map<String, Value>, String> {
    let mut entry = Map::new();

    let contig = contig(form.text("contig"), "contig")?;
    if !contig.is_empty() {
        entry.insert("contig".to_owned(), Value::String(contig));
    }
    let ligands = ligands(form.text("ligand"), "ligand")?;
    if !ligands.is_empty() {
        entry.insert("ligand".to_owned(), Value::String(ligands));
    }
    if let Some(length) = length(form.text("length"), "length")? {
        entry.insert("length".to_owned(), length);
    }
    if let Some(unindex) = selection(form.raw("unindex"), "unindex")? {
        if unindex.is_boolean() {
            return Err("unindex must be a contig string or an atom-selection object.".into());
        }
        entry.insert("unindex".to_owned(), unindex);
    }

    for name in [
        "select_fixed_atoms",
        "select_unfixed_sequence",
        "select_buried",
        "select_partially_buried",
        "select_exposed",
        "select_hbond_donor",
        "select_hbond_acceptor",
        "select_hotspots",
    ] {
        let Some(value) = selection(form.raw(name), name)? else {
            continue;
        };
        if name.starts_with("select_hbond") && !value.is_object() {
            return Err(format!(
                "{name} needs an atom-selection object, e.g. A108: ND2,CG."
            ));
        }
        if matches!(
            name,
            "select_buried" | "select_partially_buried" | "select_exposed"
        ) && value.is_boolean()
        {
            return Err(format!(
                "{name} needs a contig or an atom-selection object, not true/false."
            ));
        }
        entry.insert(name.to_owned(), value);
    }

    for name in ["symmetry", "cif_parser_args", "extra"] {
        let object = json_object(form.text(name), name)?;
        if !object.is_empty() {
            entry.insert(name.to_owned(), Value::Object(object));
        }
    }

    entry.insert(
        "dialect".to_owned(),
        Value::from(form.int("dialect", 2, 1, 2)?),
    );
    for (name, default) in [
        ("plddt_enhanced", true),
        ("redesign_motif_sidechains", false),
    ] {
        entry.insert(name.to_owned(), Value::Bool(form.boolean(name, default)?));
    }
    if form.boolean("allow_ligand_on_existing_chain", false)? {
        entry.insert(
            "allow_ligand_on_existing_chain".to_owned(),
            Value::Bool(true),
        );
    }
    if !form.text("is_non_loopy").is_empty() {
        entry.insert(
            "is_non_loopy".to_owned(),
            Value::Bool(form.boolean("is_non_loopy", true)?),
        );
    }

    let strategy = form.text("infer_ori_strategy");
    if !strategy.is_empty() {
        if !matches!(strategy, "com" | "hotspots") {
            return Err("infer_ori_strategy must be com or hotspots.".into());
        }
        entry.insert(
            "infer_ori_strategy".to_owned(),
            Value::String(strategy.to_owned()),
        );
    }
    if let Some(origin) = ori_token(form.text("ori_token"))? {
        entry.insert("ori_token".to_owned(), origin);
    }

    let input = form.text("input");
    if !input.is_empty() {
        entry.insert("input".to_owned(), Value::String(input.to_owned()));
    }
    if !form.text("partial_t").is_empty() {
        let partial = form.float("partial_t", 10.0, 0.0, f64::MAX)?;
        if input.is_empty() {
            return Err("Partial diffusion needs an input structure to start from.".into());
        }
        entry.insert("partial_t".to_owned(), number(partial));
    }

    Ok(entry)
}

/// Point every design's `input` at a file that exists on this machine.
///
/// Three spellings reach here: `"uploaded"`, meaning the structure chosen in the window; a
/// `bio-tools://rfd3/` reference, which a preset carries and which is written out of the bundle;
/// and anything else, which is a path already on disk.
fn resolve_inputs(
    mut document: Map<String, Value>,
    request: &Rfd3Request,
    run_dir: &Path,
) -> Result<Map<String, Value>, String> {
    let mut supplied_path: Option<PathBuf> = None;

    for (index, (name, entry)) in document.iter_mut().enumerate() {
        if !is_plain_name(name) {
            return Err(format!(
                "Design name \"{name}\" must use letters, digits, dots, underscores or hyphens."
            ));
        }
        let Value::Object(entry) = entry else {
            return Err(format!("Design \"{name}\" must be an object."));
        };
        let source = match entry.get("input") {
            Some(Value::String(source)) => source.clone(),
            _ => continue,
        };

        let resolved = if source == SUPPLIED_INPUT {
            let structure = request.supplied_structure.as_ref().ok_or_else(|| {
                format!(
                    "Design \"{name}\" uses the supplied structure, but none is selected in the \
                     window."
                )
            })?;
            match &supplied_path {
                Some(path) => path.clone(),
                None => {
                    let path = run_dir.join(format!("{}.pdb", file_stem(&structure.name)));
                    fs::write(&path, &structure.pdb)
                        .map_err(|error| format!("unable to write {}: {error}", path.display()))?;
                    supplied_path = Some(path.clone());
                    path
                }
            }
        } else if let Some(asset) = source.strip_prefix(BUNDLED_PREFIX) {
            let contents = presets::asset("rfd3", asset)
                .ok_or_else(|| format!("bio_tools does not bundle the structure \"{asset}\"."))?;
            let extension = if asset.ends_with(".cif") {
                "cif"
            } else {
                "pdb"
            };
            let path = run_dir.join(format!("input_{index}.{extension}"));
            fs::write(&path, contents)
                .map_err(|error| format!("unable to write {}: {error}", path.display()))?;
            path
        } else {
            let path = PathBuf::from(&source);
            if !path.is_file() {
                return Err(format!(
                    "Design \"{name}\" names an input structure that does not exist: {source}"
                ));
            }
            path
        };

        entry.insert(
            "input".to_owned(),
            Value::String(resolved.display().to_string()),
        );
    }

    Ok(document)
}

// ---------------------------------------------------------------------------------------------
// Sampler and symmetry
// ---------------------------------------------------------------------------------------------

/// The `inference_sampler.*` overrides, which are shared between designs rather than per-design.
fn sampler_options(form: &Form<'_>, symmetric: bool) -> Result<Vec<(String, String)>, String> {
    let mut options = Vec::new();

    // `cfg_t_max` has no default: left blank it is sent as null, which is how the model is told to
    // use its own. The rest have documented defaults and a floor.
    for (name, default, minimum) in [
        ("cfg_t_max", None, 0.0),
        ("cfg_scale", Some(1.5), 0.0),
        ("s_trans", Some(1.0), 0.0),
        ("noise_scale", Some(1.003), 0.0),
        ("p", Some(7.0), 0.001),
        ("gamma_min", Some(1.0), 0.0),
        ("s_jitter_origin", Some(0.0), 0.0),
    ] {
        let key = format!("inference_sampler.{name}");
        let raw = form.text(&key);
        let value = if raw.is_empty() {
            match default {
                Some(default) => number(default),
                None => Value::Null,
            }
        } else {
            number(form.float(&key, default.unwrap_or_default(), minimum, f64::MAX)?)
        };
        options.push((key, encode(&value)));
    }

    let recycle = form.text("inference_sampler.n_recycle");
    let recycle = if recycle.is_empty() {
        Value::Null
    } else {
        Value::from(form.int("inference_sampler.n_recycle", 2, 0, 1_000)?)
    };
    options.push(("inference_sampler.n_recycle".to_owned(), encode(&recycle)));

    let center = form.text("inference_sampler.center_option");
    let center = if center.is_empty() { "all" } else { center };
    if !matches!(center, "all" | "motif" | "diffuse") {
        return Err("inference_sampler.center_option must be all, motif or diffuse.".into());
    }
    options.push((
        "inference_sampler.center_option".to_owned(),
        encode(&Value::String(center.to_owned())),
    ));

    let features = string_list(
        form.raw("inference_sampler.cfg_features"),
        "inference_sampler.cfg_features",
    )?;
    for feature in &features {
        if !matches!(
            feature.as_str(),
            "active_donor" | "active_acceptor" | "ref_atomwise_rasa"
        ) {
            return Err(format!(
                "\"{feature}\" is not a classifier-free guidance feature. Use active_donor, \
                 active_acceptor, or ref_atomwise_rasa."
            ));
        }
    }
    options.push((
        "inference_sampler.cfg_features".to_owned(),
        encode(&Value::Array(
            features.into_iter().map(Value::String).collect(),
        )),
    ));

    let guidance = form.boolean("inference_sampler.use_classifier_free_guidance", false)?;
    if symmetric && guidance {
        return Err("The symmetry sampler does not support classifier-free guidance.".into());
    }
    options.push((
        "inference_sampler.use_classifier_free_guidance".to_owned(),
        encode(&Value::Bool(guidance)),
    ));
    options.push((
        "inference_sampler.allow_realignment".to_owned(),
        encode(&Value::Bool(
            form.boolean("inference_sampler.allow_realignment", false)?,
        )),
    ));

    // Symmetric and non-symmetric designs need different samplers, so the choice is the inputs' to
    // make; `auto` is what lets it be made for you.
    let expected = if symmetric { "symmetry" } else { "default" };
    let kind = form.text("inference_sampler.kind");
    let kind = if kind.is_empty() { "auto" } else { kind };
    if kind != "auto" && kind != expected {
        return Err(format!(
            "These inputs need inference_sampler.kind={expected}, or auto to choose it for you."
        ));
    }
    options.push((
        "inference_sampler.kind".to_owned(),
        encode(&Value::String(expected.to_owned())),
    ));

    Ok(options)
}

/// Whether the designs about to be run are symmetric, rejecting the combinations RFD3 cannot run.
fn symmetry_check(
    document: &Map<String, Value>,
    subset: &[String],
    form: &Form<'_>,
) -> Result<bool, String> {
    let selected: Vec<&Value> = if subset.is_empty() {
        document.values().collect()
    } else {
        subset
            .iter()
            .filter_map(|name| document.get(name))
            .collect()
    };

    let symmetry_re = Regex::new(r"^[CDcd][1-9][0-9]*$").expect("valid symmetry pattern");
    let symmetric: Vec<&Value> = selected
        .iter()
        .copied()
        .filter(|entry| entry.get("symmetry").is_some_and(|value| !value.is_null()))
        .collect();
    if symmetric.is_empty() {
        return Ok(false);
    }
    if symmetric.len() != selected.len() {
        return Err(
            "Run symmetric and non-symmetric designs separately; they use different samplers."
                .into(),
        );
    }

    for entry in symmetric {
        let symmetry = &entry["symmetry"];
        let id = symmetry
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !symmetry_re.is_match(id) {
            return Err(format!(
                "Symmetry id \"{id}\" must be a cyclic or dihedral group, such as C2 or D4."
            ));
        }
        if symmetry
            .get("is_symmetric_motif")
            .is_some_and(|value| value != &Value::Bool(true))
        {
            return Err(
                "RFD3 needs input motifs pre-symmetrized around the origin (is_symmetric_motif \
                 must be true)."
                    .into(),
            );
        }
    }

    if form.float("inference_sampler.gamma_0", 0.6, 0.0, 1.0)? <= 0.5 {
        return Err(
            "Symmetry sampling needs inference_sampler.gamma_0 above 0.5 (the default is 0.6)."
                .into(),
        );
    }
    Ok(true)
}

// ---------------------------------------------------------------------------------------------
// Field parsing
// ---------------------------------------------------------------------------------------------

/// Typed reads over the form's strings, with the bounds `bio_tools` publishes applied.
struct Form<'a> {
    values: &'a HashMap<String, String>,
}

impl<'a> Form<'a> {
    fn new(values: &'a HashMap<String, String>) -> Self {
        Self { values }
    }

    /// The value as written, for the fields whose own syntax is line- or whitespace-sensitive.
    fn raw(&self, name: &str) -> &str {
        self.values
            .get(name)
            .map(String::as_str)
            .unwrap_or_default()
    }

    fn text(&self, name: &str) -> &str {
        self.raw(name).trim()
    }

    fn boolean(&self, name: &str, default: bool) -> Result<bool, String> {
        match self.text(name).to_ascii_lowercase().as_str() {
            "" => Ok(default),
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            other => Err(format!("{name} must be true or false, not \"{other}\".")),
        }
    }

    fn int(&self, name: &str, default: i64, minimum: i64, maximum: i64) -> Result<i64, String> {
        let text = self.text(name);
        let value = if text.is_empty() {
            default
        } else {
            text.parse::<i64>()
                .map_err(|_| format!("{name} must be a whole number, not \"{text}\"."))?
        };
        if value < minimum || value > maximum {
            return Err(format!("{name} must be between {minimum} and {maximum}."));
        }
        Ok(value)
    }

    fn float(&self, name: &str, default: f64, minimum: f64, maximum: f64) -> Result<f64, String> {
        let text = self.text(name);
        let value = if text.is_empty() {
            default
        } else {
            text.parse::<f64>()
                .map_err(|_| format!("{name} must be a number, not \"{text}\"."))?
        };
        if !value.is_finite() || value < minimum || value > maximum {
            return Err(format!("{name} must be a number of at least {minimum}."));
        }
        Ok(value)
    }
}

/// Validate a comma-separated contig, returning it normalized.
///
/// A segment is a chain break (`/0`), a designed length or range (`40-120`), or a residue or range
/// taken from the input structure (`A10-25`, `M52`).
fn contig(value: &str, name: &str) -> Result<String, String> {
    if value.is_empty() {
        return Ok(String::new());
    }
    let pattern =
        Regex::new(r"^(?:/0|[A-Za-z]?[0-9]+(?:-[0-9]+)?)$").expect("valid contig pattern");
    let segments: Vec<&str> = value
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return Err(format!("{name} must contain at least one segment."));
    }
    for segment in &segments {
        if !pattern.is_match(segment) {
            return Err(format!(
                "{name} segment \"{segment}\" is not a contig segment such as A10-25, 40-120, \
                 or /0."
            ));
        }
    }
    Ok(segments.join(","))
}

/// Validate a comma-separated list of PDB chemical component IDs.
fn ligands(value: &str, name: &str) -> Result<String, String> {
    if value.is_empty() {
        return Ok(String::new());
    }
    let pattern = Regex::new(r"^[A-Za-z0-9]{1,5}$").expect("valid ligand pattern");
    let codes: Vec<&str> = value
        .split(',')
        .map(str::trim)
        .filter(|code| !code.is_empty())
        .collect();
    for code in &codes {
        if !pattern.is_match(code) {
            return Err(format!(
                "{name} entry \"{code}\" is not a PDB chemical component ID."
            ));
        }
    }
    Ok(codes.join(","))
}

/// A design length: the number or `"min-max"` string the specification takes.
fn length(value: &str, name: &str) -> Result<Option<Value>, String> {
    if value.is_empty() {
        return Ok(None);
    }
    let pattern = Regex::new(r"^[0-9]+(?:-[0-9]+)?$").expect("valid length pattern");
    if !pattern.is_match(value) {
        return Err(format!(
            "{name} must be a number or a min-max range like 140-150."
        ));
    }
    if let Some((minimum, maximum)) = value.split_once('-') {
        let minimum: u64 = minimum
            .parse()
            .map_err(|_| format!("{name} is not a range."))?;
        let maximum: u64 = maximum
            .parse()
            .map_err(|_| format!("{name} is not a range."))?;
        if minimum < 1 || minimum > maximum {
            return Err(format!(
                "{name} must be a positive, increasing length range."
            ));
        }
        return Ok(Some(Value::String(value.to_owned())));
    }
    let single: u64 = value
        .parse()
        .map_err(|_| format!("{name} is not a number."))?;
    if single < 1 {
        return Err(format!("{name} must be positive."));
    }
    Ok(Some(Value::from(single)))
}

/// Parse one of RFD3's selection fields.
///
/// Three spellings are accepted, matching the upstream documentation and the hosted form: a JSON
/// object (or boolean, where the field allows one), a bare contig such as `A10-25,B4`, and one
/// `residue: atoms` line per selection, where the atoms may be a list or one of the `ALL`, `TIP`,
/// and `BKBN` shorthands.
fn selection(raw: &str, name: &str) -> Result<Option<Value>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if trimmed.starts_with('{') || matches!(trimmed, "true" | "false") {
        let parsed: Value = serde_json::from_str(trimmed)
            .map_err(|error| format!("{name} is not valid JSON: {error}."))?;
        return match parsed {
            Value::Bool(flag) => Ok(Some(Value::Bool(flag))),
            Value::Object(object) => {
                let mut checked = Map::new();
                for (key, value) in object {
                    let atoms = value
                        .as_str()
                        .ok_or_else(|| format!("{name} must map selections to atom strings."))?;
                    checked.insert(
                        selection_key(&key, name)?,
                        Value::String(atom_list(atoms, name)?),
                    );
                }
                Ok(Some(Value::Object(checked)))
            }
            _ => Err(format!(
                "{name} must be a contig, a boolean, or an atom-selection object."
            )),
        };
    }

    if !trimmed.contains(':') {
        let joined = trimmed
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(",");
        return Ok(Some(Value::String(selection_key(&joined, name)?)));
    }

    let mut object = Map::new();
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, atoms)) = line.split_once(':') else {
            return Err(format!(
                "{name} line \"{line}\" must be \"residue: atoms\", for example \"A108: ND2,CG\"."
            ));
        };
        object.insert(
            selection_key(key, name)?,
            Value::String(atom_list(atoms, name)?),
        );
    }
    if object.is_empty() {
        return Ok(None);
    }
    Ok(Some(Value::Object(object)))
}

/// One left-hand side of a selection: a contig, or a ligand code.
fn selection_key(key: &str, name: &str) -> Result<String, String> {
    let key = key.trim();
    if key.is_empty() {
        return Err(format!("{name} has an entry with no residue or ligand."));
    }
    if let Ok(contig) = contig(key, name)
        && !contig.is_empty()
    {
        return Ok(contig);
    }
    if Regex::new(r"^[A-Za-z0-9]{1,5}$")
        .expect("valid ligand pattern")
        .is_match(key)
    {
        return Ok(key.to_owned());
    }
    Err(format!(
        "{name} entry \"{key}\" is not a residue selection such as A108 or A2-10, nor a ligand \
         code such as NAI."
    ))
}

/// One right-hand side: a shorthand, an atom list, or nothing at all.
fn atom_list(value: &str, name: &str) -> Result<String, String> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(String::new());
    }
    let upper = value.to_ascii_uppercase();
    if matches!(upper.as_str(), "ALL" | "TIP" | "BKBN") {
        return Ok(upper);
    }
    // Atom names carry primes on nucleic acids, e.g. O3'.
    let pattern = Regex::new(r"^[A-Za-z0-9']{1,6}$").expect("valid atom pattern");
    let atoms: Vec<&str> = value
        .split(',')
        .map(str::trim)
        .filter(|atom| !atom.is_empty())
        .collect();
    for atom in &atoms {
        if !pattern.is_match(atom) {
            return Err(format!("{name} atom \"{atom}\" is not an atom name."));
        }
    }
    Ok(atoms.join(","))
}

/// The origin token: three coordinates, written either bare or as a JSON array.
fn ori_token(value: &str) -> Result<Option<Value>, String> {
    if value.is_empty() {
        return Ok(None);
    }
    let inner = value
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .unwrap_or(value);
    let parts: Vec<&str> = inner
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() != 3 {
        return Err("ori_token must be three comma-separated numbers, e.g. 24,20,10.".into());
    }
    let mut coordinates = Vec::with_capacity(3);
    for part in parts {
        let coordinate: f64 = part
            .parse()
            .map_err(|_| format!("ori_token coordinate \"{part}\" is not a number."))?;
        if !coordinate.is_finite() {
            return Err("ori_token coordinates must be finite.".into());
        }
        coordinates.push(number(coordinate));
    }
    Ok(Some(Value::Array(coordinates)))
}

/// A JSON object field, which may be left blank or written as `{}`.
fn json_object(value: &str, name: &str) -> Result<Map<String, Value>, String> {
    if value.is_empty() {
        return Ok(Map::new());
    }
    match serde_json::from_str::<Value>(value) {
        Ok(Value::Object(object)) => Ok(object),
        Ok(_) => Err(format!("{name} must be a JSON object.")),
        Err(error) => Err(format!("{name} is not valid JSON: {error}.")),
    }
}

/// A list field, written either as a JSON array or as comma- or newline-separated names.
fn string_list(value: &str, name: &str) -> Result<Vec<String>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let items: Vec<String> = if trimmed.starts_with('[') {
        match serde_json::from_str::<Value>(trimmed) {
            Ok(Value::Array(items)) => items
                .into_iter()
                .map(|item| match item {
                    Value::String(text) => Ok(text),
                    _ => Err(format!("{name} must be a list of strings.")),
                })
                .collect::<Result<_, _>>()?,
            Ok(_) => return Err(format!("{name} must be a list of strings.")),
            Err(error) => return Err(format!("{name} is not valid JSON: {error}.")),
        }
    } else {
        trimmed
            .split([',', '\n'])
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_owned)
            .collect()
    };

    let mut unique = Vec::new();
    for item in items {
        if !unique.contains(&item) {
            unique.push(item);
        }
    }
    Ok(unique)
}

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

/// A finite float as JSON, preferring an integer spelling so that a whole number does not reach
/// Hydra as `3.0` where the upstream examples write `3`.
fn number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 1e15 {
        return Value::from(value as i64);
    }
    Value::from(value)
}

/// One Hydra override's right-hand side. Lists and strings are quoted for Hydra's own parser, not
/// for a shell: the command is built argument by argument and never goes through one.
fn encode(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_owned())
}

fn is_plain_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 160
        && value != "."
        && value != ".."
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// A filesystem-safe stem for a molecule that may be named anything at all.
fn file_stem(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = cleaned.trim_matches('_');
    if trimmed.is_empty() {
        "input".to_owned()
    } else {
        trimmed.chars().take(60).collect()
    }
}

fn job_name(form: &Form<'_>) -> String {
    let name = form.text("job_name");
    if is_plain_name(name) {
        name.to_owned()
    } else {
        "rfd3-design".to_owned()
    }
}

/// `<data root>/designs/rfd3/<job>-<n>`, created fresh.
///
/// Unlike the other adapters, this does not use a [`super::ToolWorkspace`]: a run takes minutes to
/// hours and the backbones it produces are the point of it, so they outlive the window rather than
/// being deleted when it closes.
fn create_run_dir(job: &str) -> io::Result<PathBuf> {
    let root = data_root()
        .ok_or_else(|| io::Error::other("unable to determine Molchanica's data directory"))?
        .join("designs")
        .join("rfd3");
    fs::create_dir_all(&root)?;

    for attempt in 1..=1_000 {
        let candidate = root.join(format!("{job}-{attempt:03}"));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "unable to allocate a run directory under {}",
            root.display()
        ),
    ))
}

/// Every backbone a run wrote, sorted so the listing is stable between runs.
fn generated_structures(run_dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, found);
                continue;
            }
            let extension = path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            // The inputs written beside the outputs are structures too; only the model's own
            // output tree is a result.
            if matches!(extension.as_str(), "pdb" | "cif") {
                found.push(path);
            }
        }
    }

    let mut found = Vec::new();
    walk(&run_dir.join("output"), &mut found);
    found.sort();
    found
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

/// The form contract `bio_tools` publishes for RFD3, loaded once.
pub fn contract() -> Result<FormContract, String> {
    FormContract::load("rfd3")
}

/// Which of a preset's values names the input mode, mapped onto [`InputMode`].
pub fn preset_mode(values: &HashMap<String, String>) -> InputMode {
    values
        .get("input_mode")
        .map(|mode| InputMode::from_contract_value(mode))
        .unwrap_or_default()
}

/// Whether a form value names a structure `bio_tools` bundles rather than a file on disk.
pub fn is_bundled_structure(value: &str) -> bool {
    value.starts_with(BUNDLED_PREFIX)
}

/// The value that points a design at the structure chosen in the window.
pub fn supplied_input_value() -> &'static str {
    SUPPLIED_INPUT
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn contigs_are_validated_and_normalized() {
        assert_eq!(
            contig("A1-80, 10, /0,B5-12", "contig").unwrap(),
            "A1-80,10,/0,B5-12"
        );
        assert!(contig("A1-80, oops", "contig").is_err());
    }

    #[test]
    fn lengths_take_a_number_or_a_range() {
        assert_eq!(length("150", "length").unwrap(), Some(Value::from(150)));
        assert_eq!(
            length("140-150", "length").unwrap(),
            Some(Value::String("140-150".to_owned()))
        );
        assert!(length("150-140", "length").is_err());
        assert_eq!(length("", "length").unwrap(), None);
    }

    #[test]
    fn selections_accept_all_three_spellings() {
        // A bare contig stays a contig, which is what the binder examples use for hotspots.
        assert_eq!(
            selection("A108,A2-10", "select_hotspots").unwrap(),
            Some(Value::String("A108,A2-10".to_owned()))
        );
        // Naming atoms promotes the whole selection to an object.
        let lines = selection("A108: ND2,CG\nNAI: ALL", "select_fixed_atoms")
            .unwrap()
            .unwrap();
        assert_eq!(lines["A108"], Value::String("ND2,CG".to_owned()));
        assert_eq!(lines["NAI"], Value::String("ALL".to_owned()));
        // JSON is what a preset carries.
        let json = selection(r#"{"IAI": ""}"#, "select_fixed_atoms")
            .unwrap()
            .unwrap();
        assert_eq!(json["IAI"], Value::String(String::new()));
    }

    #[test]
    fn ori_tokens_take_bare_or_bracketed_coordinates() {
        let bare = ori_token("24,20,10").unwrap().unwrap();
        let bracketed = ori_token("[24, 20, 10]").unwrap().unwrap();
        assert_eq!(bare, bracketed);
        assert_eq!(bare, Value::Array(vec![24.into(), 20.into(), 10.into()]));
        assert!(ori_token("24,20").is_err());
    }

    #[test]
    fn hydra_values_are_json_encoded() {
        assert_eq!(encode(&Value::Null), "null");
        assert_eq!(encode(&Value::Bool(true)), "true");
        assert_eq!(encode(&number(1.5)), "1.5");
        assert_eq!(encode(&number(3.0)), "3");
        assert_eq!(
            encode(&Value::Array(vec![Value::String("active_donor".into())])),
            r#"["active_donor"]"#
        );
    }

    #[test]
    fn the_unconditional_monomer_preset_builds_a_specification() {
        let values = form(&[("length", "150"), ("is_non_loopy", "true")]);
        let entry = native_specification(&Form::new(&values)).unwrap();
        assert_eq!(entry["length"], Value::from(150));
        assert_eq!(entry["is_non_loopy"], Value::Bool(true));
        // Documented defaults are sent explicitly, as the hosted form does.
        assert_eq!(entry["dialect"], Value::from(2));
        assert_eq!(entry["plddt_enhanced"], Value::Bool(true));
        assert!(!entry.contains_key("contig"));
    }

    #[test]
    fn partial_diffusion_needs_a_structure() {
        let values = form(&[("partial_t", "10")]);
        let error = native_specification(&Form::new(&values)).unwrap_err();
        assert!(error.contains("input structure"), "{error}");
    }

    #[test]
    fn the_symmetry_sampler_rejects_classifier_free_guidance() {
        let values = form(&[("inference_sampler.use_classifier_free_guidance", "true")]);
        let error = sampler_options(&Form::new(&values), true).unwrap_err();
        assert!(error.contains("classifier-free guidance"), "{error}");
    }

    #[test]
    fn the_sampler_kind_is_chosen_from_the_inputs() {
        let values = form(&[]);
        let options = sampler_options(&Form::new(&values), true).unwrap();
        assert!(options.contains(&(
            "inference_sampler.kind".to_owned(),
            "\"symmetry\"".to_owned()
        )));

        // Asking for the other sampler is an error rather than being silently overridden.
        let values = form(&[("inference_sampler.kind", "default")]);
        assert!(sampler_options(&Form::new(&values), true).is_err());
    }

    #[test]
    fn symmetric_and_plain_designs_cannot_share_a_run() {
        let document: Map<String, Value> =
            serde_json::from_str(r#"{"a": {"symmetry": {"id": "C2"}}, "b": {"length": 100}}"#)
                .unwrap();
        let values = form(&[]);
        let error = symmetry_check(&document, &[], &Form::new(&values)).unwrap_err();
        assert!(error.contains("separately"), "{error}");
    }

    #[test]
    fn symmetry_needs_a_high_enough_gamma() {
        let document: Map<String, Value> =
            serde_json::from_str(r#"{"a": {"symmetry": {"id": "C5"}}}"#).unwrap();
        let values = form(&[("inference_sampler.gamma_0", "0.4")]);
        let error = symmetry_check(&document, &[], &Form::new(&values)).unwrap_err();
        assert!(error.contains("gamma_0"), "{error}");

        let values = form(&[]);
        assert!(symmetry_check(&document, &[], &Form::new(&values)).unwrap());
    }

    #[test]
    fn subset_names_are_deduplicated_and_split_either_way() {
        assert_eq!(
            string_list("a, b\nc, a", "json_keys_subset").unwrap(),
            ["a", "b", "c"]
        );
        assert_eq!(
            string_list(r#"["a", "b"]"#, "json_keys_subset").unwrap(),
            ["a", "b"]
        );
    }

    #[test]
    fn every_parameter_field_in_the_contract_is_read() {
        // The point of loading the contract from bio_tools is that a field added there reaches
        // Molchanica. That only holds if this adapter actually consumes each one, so a new
        // parameter field shows up here as a failure rather than as a value silently dropped.
        let contract = FormContract::load("rfd3").unwrap();
        let handled = [
            "input",
            "contig",
            "unindex",
            "length",
            "ligand",
            "cif_parser_args",
            "extra",
            "dialect",
            "select_fixed_atoms",
            "select_unfixed_sequence",
            "select_buried",
            "select_partially_buried",
            "select_exposed",
            "select_hbond_donor",
            "select_hbond_acceptor",
            "select_hotspots",
            "redesign_motif_sidechains",
            "symmetry",
            "ori_token",
            "infer_ori_strategy",
            "plddt_enhanced",
            "is_non_loopy",
            "partial_t",
            "allow_ligand_on_existing_chain",
        ];
        for field in &contract.fields {
            if field.applies_to("parameters") && !field.applies_to("text") {
                assert!(
                    handled.contains(&field.name.as_str()),
                    "the rfd3 adapter does not read the `{}` parameter",
                    field.name
                );
            }
        }
    }
}
