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
//! Add a `--tgt herg` etc flag to specify a single target, vs every data file in the directory.

use std::{collections::HashMap, fmt::Display, io, path::Path, time::Instant};

use bio_files::md_params::ForceFieldParams;

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        DatasetTdc,
        infer::infer_general,
        train::{TGT_COL_TDC, load_training_data, train},
        train_test_split_indices::TrainTestSplit,
    },
};

/// [TDC Evaluation](https://tdcommons.ai/functions/data_evaluation/), for reference.
#[derive(Clone, Debug)]
pub struct EvalMetrics {
    /// Mean squared error.
    pub mse: f32,
    /// Root-Mean Squared deviation
    pub rmsd: f32,
    /// Mean Absolute Error
    pub mae: f32,
    /// Coefficient of determination.
    pub r2: f32,
    /// Pearson correlation coefficient.
    pub pearson: f32,
    /// Spearman correlation coefficient.
    pub spearman: f32,
    /// Area under the receiver operating characteristic. Used by TDC to score
    /// binary classifiers.
    pub auroc: Option<f32>,
}

impl Display for EvalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\nEval metrics:\n- MSE: {:.3}\n- RMSD: {:.3}\n- MAE: {:.3}\n- R²: {:.3}\n- Pearson: {:.3}\n- Spearman: {:.3}\n",
            self.mse, self.rmsd, self.mae, self.r2, self.pearson, self.spearman,
        )?;

        if let Some(v) = &self.auroc {
            write!(f, "- Auroc: {:.3}\n", v)?;
        }

        Ok(())
    }
}

fn run_infer(
    // This format matches how we load.
    data: &[(MoleculeSmall, f32)],
    data_set: DatasetTdc,
    ff_params: &ForceFieldParams,
) -> io::Result<Vec<f32>> {
    let mut result = Vec::with_capacity(data.len());

    // Models here is a cache, so we don't have to load the model for each test item.
    let mut models = HashMap::new();
    for (mol, _) in data {
        result.push(infer_general(mol, data_set, &mut models, ff_params, true)?);
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
        let end_rank = end as f32;
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

fn is_binary_01_labels(xs: &[f32]) -> bool {
    // Accept exactly 0/1 plus tiny float noise.
    const EPS: f32 = 1e-6;
    xs.iter()
        .all(|&y| (y - 0.0).abs() <= EPS || (y - 1.0).abs() <= EPS)
}

/// AUROC via rank-sum (equivalent to Mann–Whitney U).
/// `labels` is expected to be binary 0.0/1.0 (we treat >0.5 as positive).
fn auroc(scores: &[f32], labels: &[f32]) -> f32 {
    if scores.len() != labels.len() || scores.len() < 2 {
        return f32::NAN;
    }

    let mut n_pos: usize = 0;
    let mut n_neg: usize = 0;

    for &y in labels {
        if y > 0.5 {
            n_pos += 1;
        } else {
            n_neg += 1;
        }
    }

    // AUROC undefined if only one class present.
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }

    // Average ranks for ties; ranks are 1..=n (ascending by score).
    let ranks = ranks_average_ties(scores);

    let mut rank_sum_pos = 0.0f64;
    for i in 0..labels.len() {
        if labels[i] > 0.5 {
            rank_sum_pos += ranks[i] as f64;
        }
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;

    // AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    let u_pos = rank_sum_pos - (n_pos_f * (n_pos_f + 1.0) * 0.5);
    let auc = u_pos / (n_pos_f * n_neg_f);

    if auc.is_finite() {
        auc as f32
    } else {
        f32::NAN
    }
}

/// Runs training, then evaluates the model using the test set.
pub fn eval(
    data_path: &Path,
    dataset: DatasetTdc,
    tgt_col: usize,
    mol_specific_params: &mut HashMap<String, ForceFieldParams>,
    ff_params: &ForceFieldParams,
) -> io::Result<EvalMetrics> {
    let start = Instant::now();
    println!("Gathering ML metrics");

    let (csv_path, mol_path) = dataset.csv_mol_paths(data_path)?;

    // Set up training data using this train/test split. (Values otherwise might be
    // from the full set, which will overfit)

    let tts = TrainTestSplit::new(dataset);
    println!("\nTraining on the test set of len {}...\n", tts.train.len());

    // todo: Note. We are double-loading training tdata.
    // todo: Fix this, but it's minor.
    let data = load_training_data(
        &csv_path,
        &mol_path,
        tgt_col,
        &tts,
        mol_specific_params,
        ff_params,
        true,
    )?;

    // Note: This assumes the TTS logic is the same in our training pipeline. (it is for now)
    // Note: For now at least, we use the same TTS for training, and evaluation. This means
    // that we are not training on the full set. I'm unclear on how the "Validation" used in the training
    // affects this.

    if let Err(e) = train(
        data_path,
        dataset,
        TGT_COL_TDC,
        mol_specific_params,
        ff_params,
    ) {
        eprintln!("Error training {dataset}: {e}");
    }

    println!(
        "\nTraining complete. Loading test data and performing inference on set of len {}...\n",
        tts.test.len()
    );

    let inferred = run_infer(&data.test, dataset, ff_params)?;
    let tgts = data.test.iter().map(|(_, tgt)| *tgt).collect::<Vec<_>>();

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

    let auroc = if is_binary_01_labels(&tgts) {
        Some(auroc(&inferred, &tgts))
    } else {
        None
    };

    Ok(EvalMetrics {
        mse: mse as f32,
        rmsd: rmsd as f32,
        mae: mae as f32,
        r2: r2 as f32,
        pearson,
        spearman,
        auroc,
    })
}
