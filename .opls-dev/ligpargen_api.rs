//! Client for the Jorgensen group's public LigParGen service.
//! https://github.com/leelasd/LigParGen_2.3/tree/master/space

use std::io::{self, BufRead, BufReader, Cursor, Read};

use serde_json::{Value, json};

use crate::make_agent_with_timeout;

const BASE: &str = "https://lsdodda-ligpargen.hf.space";
const MAX_FILE: u64 = 8 * 1024 * 1024;

/// Generated OPLS-AA/CM1A parameters and a structure in parameter atom order.
#[derive(Clone, Debug)]
pub struct LigandParameters {
    pub itp: String,
    pub sdf: String,
    pub job_id: String,
}

fn fail(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

fn download(agent: &ureq::Agent, value: &Value) -> io::Result<Vec<u8>> {
    let url = value["url"].as_str().ok_or_else(|| fail("LigParGen omitted an output URL"))?;
    if !url.starts_with(&format!("{BASE}/gradio_api/file=")) {
        return Err(fail("LigParGen returned an unexpected output URL"));
    }
    let mut response = agent.get(url).call().map_err(|e| fail(e.to_string()))?;
    if !response.status().is_success() {
        return Err(fail(format!("LigParGen download: HTTP {}", response.status())));
    }
    let mut bytes = Vec::new();
    response.body_mut().as_reader().take(MAX_FILE + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_FILE {
        return Err(fail("LigParGen output exceeds the download limit"));
    }
    Ok(bytes)
}

/// Submit a hydrogenated MDL MOL block, preserving its protonation state and
/// geometry (optimization level zero). Neutral molecules use CM1A-LBCC;
/// charged molecules use CM1A. Blocks larger than 200 atoms are rejected by
/// the service, which also limits request frequency and supported chemistry.
/// This blocks while the job runs; applications should call it on a worker.
pub fn parameterize(mol: &str, charge: i32) -> io::Result<LigandParameters> {
    if !(-2..=2).contains(&charge) {
        return Err(fail("The public LigParGen service supports net charges from -2 to +2"));
    }
    let agent = make_agent_with_timeout(180);
    let boundary = "molchanica-ligpargen-upload";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"ligand.mol\"\r\n\
         Content-Type: chemical/x-mdl-molfile\r\n\r\n{mol}\r\n--{boundary}--\r\n"
    );
    let text = agent.post(format!("{BASE}/gradio_api/upload"))
        .header("Content-Type", format!("multipart/form-data; boundary={boundary}"))
        .send(body.as_bytes()).map_err(|e| fail(e.to_string()))?
        .body_mut().read_to_string().map_err(|e| fail(e.to_string()))?;
    let uploaded: Value = serde_json::from_str(&text).map_err(|e| fail(e.to_string()))?;
    let path = uploaded[0].as_str().ok_or_else(|| fail("LigParGen rejected the MOL upload"))?;
    let model = if charge == 0 {
        "1.14*CM1A-LBCC (neutral molecules)"
    } else {
        "1.14*CM1A (neutral or charged)"
    };
    let payload = json!({"data": ["", {
        "path": path, "orig_name": "ligand.mol", "meta": {"_type": "gradio.FileData"}
    }, "0", model, charge.to_string()]}).to_string();
    let text = agent.post(format!("{BASE}/gradio_api/call/run_ligpargen"))
        .header("Content-Type", "application/json")
        .send(payload.as_bytes()).map_err(|e| fail(e.to_string()))?
        .body_mut().read_to_string().map_err(|e| fail(e.to_string()))?;
    let submitted: Value = serde_json::from_str(&text).map_err(|e| fail(e.to_string()))?;
    let id = submitted["event_id"].as_str()
        .filter(|id| !id.is_empty() && id.chars().all(|c| c.is_ascii_hexdigit()))
        .ok_or_else(|| fail("LigParGen did not accept the job"))?;

    let mut response = agent.get(format!("{BASE}/gradio_api/call/run_ligpargen/{id}"))
        .call().map_err(|e| fail(e.to_string()))?;
    if !response.status().is_success() {
        return Err(fail(format!("LigParGen job: HTTP {}", response.status())));
    }
    let mut event = String::new();
    let mut completed = None;
    // Gradio sends heartbeat and progress events before the final result.
    for line in BufReader::new(response.body_mut().as_reader().take(MAX_FILE)).lines() {
        let line = line?;
        if let Some(kind) = line.strip_prefix("event: ") {
            event = kind.to_owned();
        } else if let Some(data) = line.strip_prefix("data: ") {
            if event == "error" {
                return Err(fail(format!("LigParGen: {data}")));
            }
            if event == "complete" {
                completed = Some(serde_json::from_str::<Value>(data).map_err(|e| fail(e.to_string()))?);
                break;
            }
        }
    }
    let result = completed.ok_or_else(|| fail("LigParGen disconnected before completing the job"))?;
    let archive = download(&agent, &result[0])?;
    let sdf = String::from_utf8(download(&agent, &result[1])?).map_err(|e| fail(e.to_string()))?;
    let mut zip = zip::ZipArchive::new(Cursor::new(archive)).map_err(|e| fail(e.to_string()))?;
    let mut itp = None;
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).map_err(|e| fail(e.to_string()))?;
        if file.name().map_err(|e| fail(e.to_string()))?.ends_with(".itp") {
            if itp.is_some() || file.size() > MAX_FILE {
                return Err(fail("Unexpected LigParGen topology archive"));
            }
            let mut text = String::new();
            (&mut file).take(MAX_FILE + 1).read_to_string(&mut text)?;
            if text.len() as u64 > MAX_FILE {
                return Err(fail("LigParGen topology exceeds the size limit"));
            }
            itp = Some(text);
        }
    }
    Ok(LigandParameters {
        itp: itp.ok_or_else(|| fail("LigParGen did not return a GROMACS topology"))?,
        sdf,
        job_id: id.to_owned(),
    })
}
