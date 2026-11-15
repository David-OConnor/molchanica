//! Running a shared-adapter tool on athanortools.com instead of locally.
//!
//! The site runs the same `bio_tools` adapters, so it takes the payload the local adapter would.
//! Its raw results are downloaded and unpacked into a fresh local results directory, beside a
//! `result.json` envelope like a local run's, so the result browser and "Open saved run…" treat
//! both alike. Any tool can run remotely, whatever this machine's OS and GPU.

use std::{
    fs, io,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use bio_apis::athanortools::{self, Client, Job, JobStatus};
use serde_json::{Value, json};

use super::{
    RunControl, Tool,
    paths::process_executables_dir,
    shared_adapter::{self, AdapterResult, collect_files, create_results_dir},
};

/// How athanortools.com records where a run came from.
const SOURCE: &str = "molchanica";

pub const DASHBOARD_URL: &str = "https://athanortools.com/dashboard/";

const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// Failed polls in a row tolerated before giving up. The job carries on on the server regardless.
const MAX_POLL_FAILURES: u32 = 12;

/// Where shared-adapter tools run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeTarget {
    #[default]
    Local,
    /// On athanortools.com.
    Remote,
}

impl ComputeTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::Local => "Local",
            Self::Remote => "Remote",
        }
    }
}

/// Run a tool on athanortools.com to completion. Blocking; call it from a worker thread.
///
/// `token` is an athanortools.com API token. Without one, the run is anonymous.
pub fn run(
    tool: Tool,
    payload: Value,
    token: Option<String>,
    control: &RunControl,
) -> io::Result<AdapterResult> {
    run_prepared(tool, token, control, |_| Ok(payload))
}

/// As [`run`], with input snapshots written into the local results directory first. See
/// [`shared_adapter::run_prepared`].
pub fn run_prepared(
    tool: Tool,
    token: Option<String>,
    control: &RunControl,
    prepare: impl FnOnce(&Path) -> io::Result<Value>,
) -> io::Result<AdapterResult> {
    let slug = shared_adapter::slug(tool);

    let directory = create_results_dir(&process_executables_dir()?, tool)?;
    let payload = prepare(&directory)?;
    fs::write(
        directory.join("submitted-form.json"),
        serde_json::to_vec_pretty(&payload)?,
    )?;

    control.set_note("Submitting to athanortools.com");
    let mut client = Client::new(token).with_source(SOURCE);
    let job = client
        .submit(slug, &payload)
        .map_err(|error| match error.kind() {
            io::ErrorKind::NotFound => io::Error::new(
                io::ErrorKind::NotFound,
                format!("athanortools.com does not offer {}", tool.spec().name()),
            ),
            _ => error,
        })?;
    let job = follow(&client, job, control)?;

    match job.status {
        JobStatus::Succeeded => (),
        JobStatus::Cancelled => {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "Run cancelled"));
        }
        _ => {
            let message = job
                .error
                .as_ref()
                .map(|error| error.message.clone())
                .unwrap_or_else(|| "The run failed".to_owned());
            return Err(with_link(message, &job));
        }
    }

    control.set_note("Downloading results from athanortools.com");
    let archive = directory.join("raw-results.zip");
    client
        .download_raw_results(&job, &mut fs::File::create(&archive)?)
        .map_err(|error| with_link(error.to_string(), &job))?;

    let unpacked = directory.join("remote");
    athanortools::extract_raw_results(&archive, &unpacked)?;

    let details = local_details(slug, &job, &unpacked)?;
    fs::write(
        directory.join("result.json"),
        serde_json::to_vec_pretty(&json!({ "result": details }))?,
    )?;

    AdapterResult::load(directory)
}

/// Poll a job until it settles, aborting it if the user cancels.
fn follow(client: &Client, mut job: Job, control: &RunControl) -> io::Result<Job> {
    let mut failures = 0;
    control.set_note(progress_note(&job));

    while !job.status.is_settled() {
        // In short steps, so an abort is acted on promptly.
        let wake = Instant::now() + POLL_INTERVAL;
        while Instant::now() < wake {
            if control.is_abort_requested() {
                let _ = client.abort(&job.id);
                return Err(io::Error::new(io::ErrorKind::Interrupted, "Run cancelled"));
            }
            thread::sleep(Duration::from_millis(200));
        }

        match client.status(&job.id) {
            Ok(polled) => {
                job = polled;
                failures = 0;
                control.set_note(progress_note(&job));
            }
            Err(error) => {
                failures += 1;
                let permanent = matches!(
                    error.kind(),
                    io::ErrorKind::PermissionDenied | io::ErrorKind::NotFound
                );
                if permanent || failures > MAX_POLL_FAILURES {
                    return Err(with_link(error.to_string(), &job));
                }
                control.set_note(format!("Retrying contact with athanortools.com · {error}"));
            }
        }
    }

    Ok(job)
}

fn progress_note(job: &Job) -> String {
    match job.status {
        JobStatus::Queued => match job.position {
            Some(ahead) if ahead > 0 => format!("Queued on athanortools.com · {ahead} ahead"),
            _ => "Queued on athanortools.com · next to run".to_owned(),
        },
        JobStatus::Running if job.cancel_requested => "Stopping on athanortools.com".to_owned(),
        JobStatus::Running => "Running on athanortools.com".to_owned(),
        JobStatus::Succeeded => "Finished on athanortools.com".to_owned(),
        JobStatus::Failed => "Failed on athanortools.com".to_owned(),
        JobStatus::Cancelled => "Cancelled on athanortools.com".to_owned(),
    }
}

fn with_link(message: String, job: &Job) -> io::Error {
    match &job.results_url {
        Some(url) => io::Error::other(format!("{message}\nRemote run: {url}")),
        None => io::Error::other(message),
    }
}

/// The site's result object, pointed at the unpacked files instead of the server's.
fn local_details(slug: &str, job: &Job, unpacked: &Path) -> io::Result<Value> {
    let run_logs = run_log_dirs(unpacked)?;
    let Some(first) = run_logs.first() else {
        return Err(with_link(
            "The downloaded results hold no outputs folder".to_owned(),
            job,
        ));
    };

    let mut output_files = Vec::new();
    for log in &run_logs {
        collect_outputs(log, &mut output_files)?;
    }
    output_files.sort();

    let mut details = match &job.result {
        Some(Value::Object(result)) => Value::Object(result.clone()),
        _ => json!({}),
    };
    let server_log_dir = details.get("run_log_dir").cloned();
    details["remote"] = json!({
        "host": athanortools::BASE_URL,
        "job_id": job.id,
        "results_url": job.results_url,
        "run_log_dir": server_log_dir,
    });
    details["tool_slug"] = json!(slug);
    details["run_log_dir"] = json!(first.to_string_lossy());
    details["output_files"] = json!(
        output_files
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
    );

    Ok(details)
}

/// The run-log directories in an unpacked archive: one folder per command the run executed, each
/// holding `outputs`, or the archive's root if it holds `outputs` itself.
fn run_log_dirs(unpacked: &Path) -> io::Result<Vec<PathBuf>> {
    if unpacked.join("outputs").is_dir() {
        return Ok(vec![unpacked.to_owned()]);
    }

    let mut dirs = Vec::new();
    for entry in fs::read_dir(unpacked)? {
        let path = entry?.path();
        if path.join("outputs").is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();

    Ok(dirs)
}

/// Files the tool created or changed under `log/outputs`, as `bio_tools`' desktop coordinator
/// selects them: audit snapshots include unchanged copies of the inputs too.
fn collect_outputs(log: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    let outputs = log.join("outputs");
    let mut found = Vec::new();
    collect_files(&outputs, &mut found)?;

    for file in found {
        let Ok(relative) = file.strip_prefix(&outputs) else {
            continue;
        };
        let input = log.join("inputs").join(relative);
        if !same_contents(&input, &file)? {
            files.push(file);
        }
    }

    Ok(())
}

fn same_contents(a: &Path, b: &Path) -> io::Result<bool> {
    if !a.is_file() || fs::metadata(a)?.len() != fs::metadata(b)?.len() {
        return Ok(false);
    }

    Ok(fs::read(a)? == fs::read(b)?)
}
