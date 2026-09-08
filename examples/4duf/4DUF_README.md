# Introduction

This example will showcase the use of various 4D models to characterize breast lesions in Ultrafast (UF) dynamic contrast enhanced (DCE) MRI based on a publication currently under review.

Three models are compared on the same task, dataset and experiment protocol using the `mindful_subream` framework. Each has its own documentation set (README, experiment, hparams and pipeline configuration) : 

| Model | Approach | Input data | Documentation |
|-------|----------|------------|---------------|
| **Ultra** | The whole 4D ultrafast sequence is input as a signle tensor into one transformer encoder (phases of `8 x 8 x 8` volume), followed by an MLP classifier. | Full 13-phase sequence and phase-ablation configurations: 7 phases, 4 phases, 3 phases, 2 middle phases (“mid”), and first and last phases only (“first-last”) | [README](./Ultra/Ultra_README.md) · [Experiment](./Ultra/Ultra_EXPERIMENT_CONFIGURATION.md) · [HParams](./Ultra/Ultra_HPARAMS_CONFIGURATION.md) · [Pipeline](./Ultra/Ultra_PIPELINE_CONFIGURATION.md) |
| **ViT-LSTM** | Each phase (`48 x 48 x 48`) is encoded by a weight shared 3D ViT, the sequence of embeddings is fused by an LSTM, followed by an MLP classifier. | Full 13-phase sequence and phase-ablation configurations: 7 phases, 4 phases, 3 phases, 2 middle phases (“mid”), and first and last phases only (“first-last”) | [README](./ViT-LSTM/ViT-LSTMa_README.md) · [Experiment](./ViT-LSTM/ViT-LSTM_EXPERIMENT_CONFIGURATION.md) · [HParams](./ViT-LSTM/ViT-LSTM_HPARAMS_CONFIGURATION.md) · [Pipeline](./ViT-LSTM/ViT-LSTM_PIPELINE_CONFIGURATION.md) |
| **ViT-Transformer** | Each phase (`48 x 48 x 48`) is encoded by a weight shared 3D ViT, the sequence of embeddings is fused by a Transformer, followed by an MLP classifier. | Full 13-phase sequence and phase-ablation configurations: 7 phases, 4 phases, 3 phases, 2 middle phases (“mid”), and first and last phases only (“first-last”) | [README](./ViT-Transformer/ViT-Transformer-LSTMa_README.md) · [Experiment](./ViT-Transformer/ViT-Transformer_EXPERIMENT_CONFIGURATION.md) · [HParams](./ViT-Transformer/ViT-Transformer_HPARAMS_CONFIGURATION.md) · [Pipeline](./ViT-Transformer/ViT-Transformer_PIPELINE_CONFIGURATION.md) |



## Tasks and datasets

The classification of lesions targets three **classification tasks**, each with its own folds:

- **B-G-M**: Benign vs. lymph nodes vs. malignant (3 classes),
- **BG-M**: Benign + lymph nodes vs. malignant (2 classes),
- **B-M**: Benign vs. malignant (2 classes, lymph nodes excluded)


datasets-->


## Common experimental protocol

- **5-fold cross-validation** (`folds: "all"`), repeated over **5 fixed seeds**,
- **imbalanced sampler** (`use_imbalanced_sampler: "v2"`) to mitigate class imbalance,
- Adam optimizer (`lr = 1e-4`), batch size 64, up to 500 epochs with early stopping on `validation_loss_epoch`,
- a **phase-count study**: each model is trained with the full sequence (13 phases) and with less phase versions (7, 4, 3 and 2 phases — middle or first/last), keeping the encoder hparams and the pipeline consistent.


## Running experiment

All experiments can be run with the following command line: 

```
python -m mindful_subream.main --config <exp_data>/config/runs/<run_file>.json
```

You must run it in the folder containing the folder `mindful_subream` (which contains `main.py`). 

Only the experiments whose `skip` flag is set to `no` are executed. Outputs (tensorboard files, `.ckpt` checkpoints, `formatted_summary.csv`, occlusion maps) are written to `<exp_data>/logs/<experiment_name>`. 

## Repository layout
 
```
<exp_data>/config/
├── 4D_ultra/
│   ├── models/       # ultra encoder hparams (per phase count) + MLP classifier hparams
│   ├── pipelines/    # 4D tensor pipelines (phase selection via filter_slices)
│   └── runs/         # run configuration file
├── ViT_LSTM/
│   ├── models/       # per-phase ViT + LSTM fusion hparams + MLP classifier hparams
│   ├── pipelines/    # per-phase modality pipelines (take_slice per phase)
│   └── runs/         # run configuration file
└── ViT_Transformer/
│   ├── models/       # per-phase ViT + Transformer fusion hparams + MLP classifier hparams
│   ├── pipelines/    # per-phase modality pipelines (take_slice per phase)
│   └── runs/         # run configuration file
```
