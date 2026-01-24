//! Evaluate the performance of a model.'
//!
//! [TDC scaffold split](https://tdcommons.ai/functions/data_split/):
//!
//! Example getting indices (or SMILES?) of the train/test split usinb the recommended
//! TDC Python functions. (Note: TDC may not be Windows compatible.)
//!
//! See `scripts/train_test_split.py` for an example of how to use this.
//!
//! You can run the eval fns directly, or call the `train` executable with the `--eval` param:
//! `cargo r --release --features train --bin train -- --path C:/Users/the_a/Desktop/bio_misc/tdc_data --eval`
//!
//! Add a `--tgt herg` etc flag to spsecify a single target, vs every data file in the directory.

use std::{collections::HashMap, fmt::Display, io, path::Path, str::FromStr, time::Instant};

use bio_files::md_params::ForceFieldParams;

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        infer::infer_general,
        train::{load_training_data, train_on_path},
        train_test_split_indices::TrainTestSplit,
    },
};

/// [TDC Evaluation](https://tdcommons.ai/functions/data_evaluation/), for reference.
#[derive(Clone, Debug)]
pub struct EvalMetrics {
    /// Mean squared error.
    pub mse: f32,
    /// Root-Mean Squared error
    pub rmse: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Coefficient of determination.
    pub r2: f32,
    /// Pearson correlation coefficient.
    pub pearson: f32,
    /// Spearman correlation coefficient.
    pub spearman: f32,
    // todo More A/R
}

impl Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\nEval metrics:\n- MSE: {:.3}\n- RMSE: {:.3}\n- MAE: {:.3}\n- R²: {:.3}\n- Pearson: {:.3}\n- Spearman: {:.3}\n",
            self.mse, self.rmse, self.mae, self.r2, self.pearson, self.spearman
        )
    }
}

fn run_infer(
    // This format matches how we load.
    data: &[(MoleculeSmall, f32)],
    data_set: DatasetTdc,
) -> io::Result<Vec<f32>> {
    let mut result = Vec::with_capacity(data.len());

    // Models here is a cache, so we don't have to load the model for each test item.
    let mut models = HashMap::new();
    for (mol, _) in data {
        result.push(infer_general(mol, data_set, &mut models)?);
    }

    Ok(result)
}

fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NAN;
    }
    xs.iter().copied().sum::<f32>() / (xs.len() as f32)
}

fn pearson_corr(xs: &[f32], ys: &[f32]) -> f32 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f32::NAN;
    }

    let mx = mean(xs);
    let my = mean(ys);

    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    let mut sxy = 0.0f32;

    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
    }

    let denom = (sxx * syy).sqrt();
    if denom == 0.0 || !denom.is_finite() {
        return f32::NAN;
    }
    sxy / denom
}

fn ranks_average_ties(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();

    idx.sort_by(|&a, &b| {
        xs[a]
            .partial_cmp(&xs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranks = vec![0.0f32; n];
    let mut i = 0usize;

    while i < n {
        let start = i;
        let v = xs[idx[i]];
        i += 1;

        while i < n && xs[idx[i]] == v {
            i += 1;
        }

        let end = i; // exclusive
        let start_rank = (start as f32) + 1.0;
        let end_rank = (end as f32);
        let avg_rank = (start_rank + end_rank) * 0.5;

        for j in start..end {
            ranks[idx[j]] = avg_rank;
        }
    }

    ranks
}

fn spearman_corr(xs: &[f32], ys: &[f32]) -> f32 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return f32::NAN;
    }
    let rx = ranks_average_ties(xs);
    let ry = ranks_average_ties(ys);
    pearson_corr(&rx, &ry)
}

pub fn eval(
    csv_path: &Path,
    sdf_path: &Path,
    tgt_col: usize,
    mol_specific_params: &mut HashMap<String, ForceFieldParams>,
    gaff2: &ForceFieldParams,
) -> io::Result<EvalMetrics> {
    let start = Instant::now();
    println!("Gathering ML metrics");

    let target_name = match csv_path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return Err(io::Error::new(io::ErrorKind::Other, "Invalid CSV path")),
    };

    let dataset = DatasetTdc::from_str(target_name)?;
    let tts = TrainTestSplit::new(dataset);

    // todo: Hard-coded for now.
    let path = Path::new("C:/Users/the_a/Desktop/bio_misc/tdc_data");

    // Set up training data using this train/test split. (Values otherwise might be
    // from the full set, which will overfit)
    println!("\nTraining on the test set of len {}...\n", tts.train.len());

    // Note: For now at least, we use the same TTS for training, and evaluation. This means
    // that we are not training on the full set. I'm unclear on how the "Validation" used in the training
    // affects this.
    train_on_path(path, Some(dataset), false, mol_specific_params, gaff2);

    let loaded = load_training_data(
        csv_path,
        sdf_path,
        tgt_col,
        Some(&tts.test),
        mol_specific_params,
        gaff2,
    )?;

    println!(
        "\nTraining complete. Loading test data and performing inference on set of len {}...\n",
        loaded.len()
    );

    let inferred = run_infer(&loaded, dataset)?;
    let tgts = loaded.iter().map(|(_, tgt)| *tgt).collect::<Vec<_>>();

    if tgts.len() != inferred.len() || tgts.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Inferred length mismatch",
        ));
    }

    let n = tgts.len() as f64;

    // Accumulate as f64
    let mut se_sum = 0.0f64;
    let mut ae_sum = 0.0f64;

    for i in 0..tgts.len() {
        // todo: FOr now.
        if i < 30 {
            println!("Expected: {}, Inferred: {}", tgts[i], inferred[i]);
        }

        let err = (inferred[i] - tgts[i]) as f64;
        se_sum += err * err;
        ae_sum += err.abs();
    }

    let mse = se_sum / n;
    let rmsd = mse.sqrt();
    let mae = ae_sum / n;

    let y_mean = mean(&tgts);
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;

    for i in 0..tgts.len() {
        let y = tgts[i] as f64;
        let yhat = inferred[i] as f64;
        let r = y - yhat;
        ss_res += r * r;

        let d = y - y_mean as f64;
        ss_tot += d * d;
    }

    let r2 = if ss_tot == 0.0 || !ss_tot.is_finite() {
        f64::NAN
    } else {
        1.0 - (ss_res / ss_tot)
    };

    let pearson = pearson_corr(&inferred, &tgts);
    let spearman = spearman_corr(&inferred, &tgts);

    let elapsed = start.elapsed();
    println!("ML metrics gathered in {:?}", elapsed);

    Ok(EvalMetrics {
        mse: mse as f32,
        rmse: rmsd as f32,
        mae: mae as f32,
        r2: r2 as f32,
        pearson,
        spearman,
    })
}
