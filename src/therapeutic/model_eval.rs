//! Evaluate the performance of a model.

use std::{collections::HashMap, fs, io, path::Path, time::Instant};

use bio_files::Sdf;

use crate::{
    molecules::small::MoleculeSmall,
    therapeutic::{
        infer::{Infer, infer_general},
        train::{TGT_COL_TDC, load_training_data},
    },
};

/// [TDC Evaluation](https://tdcommons.ai/functions/data_evaluation/), for reference.
#[derive(Clone, Debug)]
pub struct EvalMetrics {
    /// Mean squared error.
    pub mse: f32,
    /// Root-Mean Squared error
    pub rmsd: f32,
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

fn run_infer(
    // mols: &[MoleculeSmall],
    // This format matches how we load.
    data: &[(MoleculeSmall, f32)],
    target_name: &str,
    models: &mut HashMap<String, Infer>,
) -> io::Result<Vec<f32>> {
    let mut result = Vec::with_capacity(data.len());

    for (mol, _) in data {
        result.push(infer_general(mol, target_name, models)?);
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
    target_name: &str,
    csv_path: &Path,
    sdf_path: &Path,
    limit: usize,
    models: &mut HashMap<String, Infer>,
) -> io::Result<EvalMetrics> {
    let start = Instant::now();
    println!("Gathering ML metrics");

    let tgt_col = TGT_COL_TDC;

    let mut loaded = load_training_data(csv_path, sdf_path, tgt_col)?;
    if limit != 0 && loaded.len() > limit {
        loaded.truncate(limit);
    }

    let inferred = run_infer(&loaded, target_name, models)?;
    let tgts = loaded.iter().map(|(_, tgt)| *tgt).collect::<Vec<_>>();

    if tgts.len() != inferred.len() || tgts.is_empty() {
        // todo: No...
        return Ok(EvalMetrics {
            mse: f32::NAN,
            rmsd: f32::NAN,
            mae: f32::NAN,
            r2: f32::NAN,
            pearson: f32::NAN,
            spearman: f32::NAN,
        });
    }

    let n = tgts.len() as f32;

    let mut se_sum = 0.;
    let mut ae_sum = 0.;

    for i in 0..tgts.len() {
        // todo: FOr now.
        if i < 30 {
            println!("Expected: {}, Inferred: {}", tgts[i], inferred[i]);
        }

        let err = inferred[i] - tgts[i];
        se_sum += err * err;
        ae_sum += err.abs();
    }

    let mse = se_sum / n;
    let rmsd = mse.sqrt();
    let mae = ae_sum / n;

    let y_mean = mean(&tgts);
    let mut ss_res = 0.0f32;
    let mut ss_tot = 0.0f32;

    for i in 0..tgts.len() {
        let y = tgts[i];
        let yhat = inferred[i];
        let r = y - yhat;
        ss_res += r * r;

        let d = y - y_mean;
        ss_tot += d * d;
    }

    let r2 = if ss_tot == 0.0 || !ss_tot.is_finite() {
        f32::NAN
    } else {
        1.0 - (ss_res / ss_tot)
    };

    let pearson = pearson_corr(&inferred, &tgts);
    let spearman = spearman_corr(&inferred, &tgts);

    let elapsed = start.elapsed();
    println!("ML metrics gathered in {:?}", elapsed);

    Ok(EvalMetrics {
        mse,
        rmsd,
        mae,
        r2,
        pearson,
        spearman,
    })
}
