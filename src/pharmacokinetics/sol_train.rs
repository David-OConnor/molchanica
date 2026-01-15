use std::{fs, io, path::Path};

use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::Module,
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        Tensor, TensorData, activation,
        backend::{AutodiffBackend, Backend},
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};

// use crate::pharmacokinetics::sol_infer::{
// use molchanica::pharmacokinetics::sol_infer::{
//     AQ_SOL_FEATURE_DIM, AqSolModel, AqSolModelConfig, MODEL_CFG_FILE, MODEL_FILE, SCALER_FILE,
//     StandardScaler,
// };

// We have put the constants here in the training module; we are having trouble importing from
// Molchanica directly.
pub const AQ_SOL_FEATURE_DIM: usize = 17;

pub const MODEL_CFG_FILE: &str = "aqsol_model_config.json";
pub const MODEL_FILE: &str = "aqsol_model";
pub const SCALER_FILE: &str = "aqsol_scaler.json";

type TrainBackend = Autodiff<NdArray>;
type TrainDevice = <TrainBackend as Backend>::Device;

type ValidBackend = <TrainBackend as AutodiffBackend>::InnerBackend; // == NdArray
type ValidDevice = <ValidBackend as Backend>::Device;

#[derive(Config, Debug)]
pub struct AqSolModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub hidden_dim2: usize,
}

#[derive(Module, Debug)]
pub struct AqSolModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl AqSolModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AqSolModel<B> {
        AqSolModel {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim2).init(device),
            fc3: LinearConfig::new(self.hidden_dim2, 1).init(device),
        }
    }
}

impl<B: Backend> AqSolModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x); // Burn 0.19: relu is a free function, not a method. :contentReference[oaicite:2]{index=2}
        let x = self.fc2.forward(x);
        let x = activation::relu(x);
        self.fc3.forward(x)
    }
}

impl TrainStep<AqSolBatch<TrainBackend>, RegressionOutput<TrainBackend>>
    for AqSolModel<TrainBackend>
{
    fn step(&self, batch: AqSolBatch<TrainBackend>) -> TrainOutput<RegressionOutput<TrainBackend>> {
        let pred = self.forward(batch.x);
        let loss = MseLoss::new().forward(pred.clone(), batch.y.clone(), Reduction::Mean);

        let grads = loss.backward();
        let output = RegressionOutput::new(loss, pred, batch.y);

        TrainOutput::new(self, grads, output)
    }
}

impl ValidStep<AqSolBatch<ValidBackend>, RegressionOutput<ValidBackend>>
    for AqSolModel<ValidBackend>
{
    fn step(&self, batch: AqSolBatch<ValidBackend>) -> RegressionOutput<ValidBackend> {
        let pred = self.forward(batch.x);
        let loss = MseLoss::new().forward(pred.clone(), batch.y.clone(), Reduction::Mean);
        RegressionOutput::new(loss, pred, batch.y)
    }
}

const MODEL_DIR: &str = "ml_models/aqsol"; // if you want absolute, keep absolute; just use Path::new below.
const AQ_SOL_DB_CSV_PATH: &str = "C:/Users/the_a/Desktop/bio_misc/AqSolDB/data/AqSolDb.csv";

// Training config
const BATCH_SIZE: usize = 128;
const EPOCHS: usize = 80;
const HIDDEN_DIM: usize = 128;
const HIDDEN_DIM2: usize = 64;
const LR: f64 = 3e-4; // Burn 0.19 LearnerBuilder expects f64 scheduler, not f32.
const SEED: u64 = 42;
const TRAIN_SPLIT: f32 = 0.9;

// ...

#[derive(Clone, Debug)]
struct Sample {
    x: [f32; AQ_SOL_FEATURE_DIM],
    y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardScaler {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Clone, Debug)]
struct AqSolBatch<B: Backend> {
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
}

#[derive(Clone)]
struct AqSolBatcher {
    scaler: StandardScaler,
}

#[derive(Clone, Debug, Deserialize)]
struct CsvRow {
    #[serde(rename = "ID")]
    id: String,

    #[serde(rename = "Solubility")]
    solubility: f32,

    #[serde(rename = "MolWt")]
    mol_weight: f32,
    #[serde(rename = "MolLogP")]
    mol_log_p: f32,
    #[serde(rename = "MolMR")]
    mol_mr: f32,

    #[serde(rename = "HeavyAtomCount")]
    heavy_atom_count: u16,
    #[serde(rename = "NumHAcceptors")]
    num_h_acceptors: u8,
    #[serde(rename = "NumHDonors")]
    num_h_donors: u8,
    #[serde(rename = "NumHeteroatoms")]
    num_het_atoms: u8,
    #[serde(rename = "NumRotatableBonds")]
    num_rotatable_bonds: u8,
    #[serde(rename = "NumValenceElectrons")]
    num_valence_elec: u16,

    #[serde(rename = "NumAromaticRings")]
    num_aromatic_rings: u8,
    #[serde(rename = "NumSaturatedRings")]
    num_saturated_rings: u8,
    #[serde(rename = "NumAliphaticRings")]
    num_aliphatic_rings: u8,
    #[serde(rename = "RingCount")]
    ring_count: u8,

    #[serde(rename = "TPSA")]
    tpsa: f32,
    #[serde(rename = "LabuteASA")]
    labute_asa: f32,
    #[serde(rename = "BalabanJ")]
    balaban_j: f32,
    #[serde(rename = "BertzCT")]
    bertz_ct: f32,
}

fn features_from_csv_row(r: &CsvRow) -> [f32; AQ_SOL_FEATURE_DIM] {
    [
        r.mol_weight,
        r.mol_log_p,
        r.mol_mr,
        r.heavy_atom_count as f32,
        r.num_h_acceptors as f32,
        r.num_h_donors as f32,
        r.num_het_atoms as f32,
        r.num_rotatable_bonds as f32,
        r.num_valence_elec as f32,
        r.num_aromatic_rings as f32,
        r.num_saturated_rings as f32,
        r.num_aliphatic_rings as f32,
        r.ring_count as f32,
        r.tpsa,
        r.labute_asa,
        r.balaban_j,
        r.bertz_ct,
    ]
}

fn read_csv(path: &Path) -> io::Result<Vec<Sample>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    let mut out = Vec::new();
    for result in rdr.deserialize::<CsvRow>() {
        let row = result.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        out.push(Sample {
            x: features_from_csv_row(&row),
            y: row.solubility,
        });
    }
    Ok(out)
}

impl<B: Backend> Batcher<B, Sample, AqSolBatch<B>> for AqSolBatcher {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> AqSolBatch<B> {
        let bs = items.len();

        let mut x_all = Vec::with_capacity(bs * AQ_SOL_FEATURE_DIM);
        let mut y_all = Vec::with_capacity(bs);

        for mut s in items {
            apply_scaler_in_place(&self.scaler, &mut s.x);
            x_all.extend_from_slice(&s.x);
            y_all.push(s.y);
        }

        // Burn 0.19: build Tensor from TensorData::new(vec, shape) :contentReference[oaicite:6]{index=6}
        let x_data = TensorData::new(x_all, [bs, AQ_SOL_FEATURE_DIM]);
        let y_data = TensorData::new(y_all, [bs, 1usize]);

        let x = Tensor::<B, 2>::from_data(x_data, device);
        let y = Tensor::<B, 2>::from_data(y_data, device);

        AqSolBatch { x, y }
    }
}

fn apply_scaler_in_place(scaler: &StandardScaler, x: &mut [f32; AQ_SOL_FEATURE_DIM]) {
    for i in 0..AQ_SOL_FEATURE_DIM {
        let s = if scaler.std[i].abs() < 1.0e-12 {
            1.0
        } else {
            scaler.std[i]
        };
        x[i] = (x[i] - scaler.mean[i]) / s;
    }
}

fn fit_scaler(train: &[Sample]) -> StandardScaler {
    let n = train.len().max(1) as f32;

    let mut mean = vec![0.0f32; AQ_SOL_FEATURE_DIM];
    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            mean[i] += s.x[i];
        }
    }
    for i in 0..AQ_SOL_FEATURE_DIM {
        mean[i] /= n;
    }

    let mut var = vec![0.0f32; AQ_SOL_FEATURE_DIM];
    for s in train {
        for i in 0..AQ_SOL_FEATURE_DIM {
            let d = s.x[i] - mean[i];
            var[i] += d * d;
        }
    }
    for i in 0..AQ_SOL_FEATURE_DIM {
        var[i] /= n;
    }

    let std = var.into_iter().map(|v| v.sqrt().max(1.0e-6)).collect();
    StandardScaler { mean, std }
}

pub fn main() {
    let model_dir = Path::new(MODEL_DIR);
    fs::create_dir_all(model_dir).unwrap();

    let mut all = read_csv(Path::new(AQ_SOL_DB_CSV_PATH)).unwrap();
    if all.is_empty() {
        eprintln!("No rows loaded from CSV");
        return;
    }

    let mut rng = StdRng::seed_from_u64(SEED);
    all.shuffle(&mut rng);

    let split = ((all.len() as f32) * TRAIN_SPLIT).round() as usize;
    let split = split.clamp(1, all.len() - 1);
    let (train_raw, valid_raw) = all.split_at(split);

    let scaler = fit_scaler(train_raw);
    fs::write(
        model_dir.join(SCALER_FILE),
        serde_json::to_vec_pretty(&scaler).unwrap(),
    )
    .unwrap();

    let model_cfg = AqSolModelConfig {
        input_dim: AQ_SOL_FEATURE_DIM,
        hidden_dim: HIDDEN_DIM,
        hidden_dim2: HIDDEN_DIM2,
    };
    fs::write(
        model_dir.join(MODEL_CFG_FILE),
        serde_json::to_vec_pretty(&model_cfg).unwrap(),
    )
    .unwrap();

    let train_ds = InMemDataset::new(train_raw.to_vec());
    let valid_ds = InMemDataset::new(valid_raw.to_vec());

    let device_train = TrainDevice::default();
    let device_valid = ValidDevice::default();

    let train_loader =
        DataLoaderBuilder::<TrainBackend, Sample, AqSolBatch<TrainBackend>>::new(AqSolBatcher {
            scaler: scaler.clone(),
        })
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .build(train_ds);

    let valid_loader =
        DataLoaderBuilder::<ValidBackend, Sample, AqSolBatch<ValidBackend>>::new(AqSolBatcher {
            scaler,
        })
        .batch_size(BATCH_SIZE)
        .build(valid_ds);

    let model = model_cfg.init::<TrainBackend>(&device_train);

    let optim = AdamConfig::new().init();
    let recorder = CompactRecorder::new();

    let learner = burn::train::LearnerBuilder::new(model_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(recorder.clone())
        .build(model, optim, LR);

    let trained = learner.fit(train_loader, valid_loader);

    let trained_model = trained.model;
    trained_model
        .save_file(model_dir.join(MODEL_FILE), &recorder)
        .unwrap();
}
