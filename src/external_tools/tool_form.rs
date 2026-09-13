//! Form contracts and presets, loaded from `bio_tools` at runtime.
//!
//! `bio_tools` ships one JSON file per tool describing every input that tool takes — name, label,
//! kind, default, bounds, help text, and which input mode it belongs to — plus a list of worked
//! presets. `bio_web` renders its pages straight out of those files. Reading the same files here
//! means Molchanica's tool windows offer the same fields, with the same labels, help, bounds, and
//! defaults, and that adding a field upstream shows up in both without either restating the
//! other's list.
//!
//! Values are held as strings throughout, exactly as an HTML form would submit them. Interpreting
//! them — a contig, a length range, an atom selection — belongs to `bio_tools`' Python adapter for
//! the tool, which is where the tool's own rules live; see [`super::shared_adapter`].

use std::collections::{BTreeMap, HashMap};

use bio_tools::tool_definitions::{fields, presets};
use serde::Deserialize;
use serde_json::Value;

/// What widget a field is drawn with. Mirrors the `kind` strings `bio_tools` uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldKind {
    Text,
    TextArea,
    Number,
    Checkbox,
    Select,
    /// A path to a file on disk. `bio_web` uploads these; Molchanica points at them in place.
    File,
    Molecules,
    /// A widget the web app draws itself, or a `kind` added upstream after this was written.
    /// Rendered as a plain text box rather than dropped, so a new field is still reachable.
    Other,
}

impl FieldKind {
    fn parse(value: &str) -> Self {
        match value {
            "text" => Self::Text,
            "textarea" => Self::TextArea,
            "number" => Self::Number,
            "checkbox" => Self::Checkbox,
            "select" => Self::Select,
            "file" => Self::File,
            "molecule_builder" => Self::Molecules,
            _ => Self::Other,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct FieldOption {
    pub value: String,
    pub label: String,
}

/// One input, as `bio_tools` describes it.
#[derive(Clone, Debug, Deserialize)]
pub struct FormField {
    pub name: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    pub default: Value,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub help: String,
    #[serde(default)]
    pub options: Vec<FieldOption>,
    /// Preferred height of a `textarea`, in rows.
    #[serde(default)]
    pub rows: Option<usize>,
    #[serde(default)]
    pub minimum: Option<f64>,
    #[serde(default)]
    pub maximum: Option<f64>,
    #[serde(default)]
    pub step: Value,
    /// Comma-separated file extensions a `file` field accepts, e.g. `.pdb,.cif`.
    #[serde(default)]
    pub accept: String,
    /// Which [`FormContract::field_groups`] heading this belongs under.
    #[serde(default)]
    pub group: String,
    /// Comma-separated input modes this field applies to. Empty means every mode.
    #[serde(default)]
    pub input_modes: String,
    /// How the hosted form adapts the upstream input, where it has to. Molchanica runs the tool
    /// locally and mostly does not, so this is shown as a footnote rather than as the help text.
    #[serde(default)]
    pub help_note: String,
    #[serde(default)]
    pub task: String,
    #[serde(default)]
    pub molecule_features: String,
    #[serde(default)]
    pub managed_by_runner: bool,
}

impl FormField {
    pub fn kind(&self) -> FieldKind {
        FieldKind::parse(&self.kind)
    }

    /// Whether this field is shown in `mode`. A field with no modes listed applies to all of them.
    pub fn applies_to(&self, mode: &str) -> bool {
        self.input_modes.trim().is_empty()
            || self
                .input_modes
                .split(',')
                .any(|listed| listed.trim() == mode)
    }

    pub fn applies_to_task(&self, task: &str) -> bool {
        self.task.trim().is_empty() || self.task.split(',').any(|listed| listed.trim() == task)
    }

    /// The default, in the string form the widgets and the adapter both work in.
    pub fn default_text(&self) -> String {
        value_to_text(&self.default)
    }

    /// The file extensions [`FieldKind::File`] accepts, without their leading dots.
    pub fn accepted_extensions(&self) -> Vec<String> {
        self.accept
            .split(',')
            .map(|extension| extension.trim().trim_start_matches('.'))
            .filter(|extension| !extension.is_empty())
            .map(str::to_owned)
            .collect()
    }
}

/// A heading the fields are grouped under, with the upstream documentation for that group.
#[derive(Clone, Debug, Deserialize)]
pub struct FieldGroup {
    pub label: String,
    #[serde(default)]
    pub docs_url: Option<String>,
}

/// The selector that decides which subset of the fields applies, e.g. RFD3's
/// "set parameters here" versus "enter a JSON document".
#[derive(Clone, Debug, Deserialize)]
pub struct InputModeSelector {
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub default: String,
    #[serde(default)]
    pub options: Vec<FieldOption>,
}

/// Everything `bio_tools` says about one tool's inputs.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct FormContract {
    #[serde(default)]
    pub tasks: Vec<FieldOption>,
    #[serde(default)]
    pub fields: Vec<FormField>,
    #[serde(default)]
    pub field_groups: Vec<FieldGroup>,
    /// Absent for tools whose fields all apply at once.
    #[serde(default)]
    pub input_modes: Option<InputModeSelector>,
    /// Upstream documentation links, keyed by a short name.
    #[serde(default)]
    pub references: BTreeMap<String, String>,
}

impl FormContract {
    /// Load and parse the contract `bio_tools` publishes for `slug`.
    pub fn load(slug: &str) -> Result<Self, String> {
        let text = fields::by_slug(slug)
            .ok_or_else(|| format!("bio_tools has no form contract for `{slug}`"))?;
        serde_json::from_str(text).map_err(|error| {
            format!("the bio_tools form contract for `{slug}` is unreadable: {error}")
        })
    }

    pub fn field(&self, name: &str) -> Option<&FormField> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Every field's default, keyed by field name, plus the first task where the tool has tasks.
    /// What a fresh form, or one a preset is about to be loaded over, starts from.
    pub fn defaults(&self) -> HashMap<String, String> {
        let mut values: HashMap<String, String> = self
            .fields
            .iter()
            .map(|field| (field.name.clone(), field.default_text()))
            .collect();

        if let Some(task) = self.tasks.first() {
            values.insert("task".into(), task.value.clone());
        }
        values
    }

    /// The field a structure file goes in, in `mode`: the first file field that accepts PDB.
    ///
    /// This is where the tool window puts a structure written from an opened molecule, so that
    /// no tool's field names have to be known outside its contract. `None` for tools that take no
    /// structure.
    pub fn structure_field(&self, mode: &str) -> Option<&FormField> {
        self.fields.iter().find(|field| {
            field.kind() == FieldKind::File
                && field.applies_to(mode)
                && field
                    .accepted_extensions()
                    .iter()
                    .any(|extension| extension.eq_ignore_ascii_case("pdb"))
        })
    }

    /// The input mode to start in.
    pub fn default_mode(&self) -> String {
        self.input_modes
            .as_ref()
            .map(|modes| modes.default.clone())
            .unwrap_or_default()
    }

    /// The groups the fields visible in `mode` fall into, in the order `bio_tools` lists them, each
    /// with its heading, the declared group of that name if there is one, and its fields. A group
    /// with nothing visible in this mode is left out entirely.
    ///
    /// The heading is the `group` the fields themselves carry, which is what `bio_web`'s form
    /// groups by as well; `field_groups` names only the subset of those groups that have upstream
    /// documentation to link, so a tool that declares none still gets one section per group its
    /// fields name. Taking the heading from `field_groups` instead collapsed every group of such a
    /// tool into one unnamed section, which then drew several identically-named headers.
    pub fn groups_in_mode(&self, mode: &str) -> Vec<(&str, Option<&FieldGroup>, Vec<&FormField>)> {
        let mut ordered: Vec<&str> = self
            .field_groups
            .iter()
            .map(|group| group.label.as_str())
            .collect();
        for field in &self.fields {
            if !ordered.contains(&field.group.as_str()) {
                ordered.push(&field.group);
            }
        }

        ordered
            .into_iter()
            .filter_map(|label| {
                let members: Vec<&FormField> = self
                    .fields
                    .iter()
                    .filter(|field| field.group == label && field.applies_to(mode))
                    .collect();
                if members.is_empty() {
                    return None;
                }
                let group = self.field_groups.iter().find(|group| group.label == label);
                Some((label, group, members))
            })
            .collect()
    }
}

/// One worked example from `bio_tools`, as a set of form values to load.
#[derive(Clone, Debug, Deserialize)]
pub struct Preset {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub source_url: Option<String>,
    values: BTreeMap<String, Value>,
}

impl Preset {
    /// Every preset `bio_tools` publishes for `slug`, in the order it lists them. Tools with no
    /// presets yield an empty list rather than an error.
    pub fn load_all(slug: &str) -> Result<Vec<Self>, String> {
        let Some(text) = presets::by_slug(slug) else {
            return Ok(Vec::new());
        };
        serde_json::from_str(text)
            .map_err(|error| format!("the bio_tools presets for `{slug}` are unreadable: {error}"))
    }

    /// This preset's values, in the string form the widgets work in.
    pub fn form_values(&self) -> HashMap<String, String> {
        self.values
            .iter()
            .map(|(name, value)| (name.clone(), value_to_text(value)))
            .collect()
    }
}

/// Render one JSON value as the text a form field would hold.
///
/// Objects and arrays keep their JSON spelling — several RFD3 fields are documented as JSON, and
/// the adapter parses them back — while scalars are written plainly so that a number does not
/// arrive in a text box wrapped in quotes.
pub fn value_to_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => text.clone(),
        Value::Array(_) | Value::Object(_) => {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }
    }
}
