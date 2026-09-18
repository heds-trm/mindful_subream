# Model HParams Configuration
 
This configuration defines the composite classification model used in the [4DUF example](../4DUF_README.md). The model is made of two parts, each configured by its own hparams file:

- an **MLP classifier** (classifier model, `class_name: "mlp"`) for the final prediction,
- a **ViT-LSTM encoder** (sub-module `representation_model`, `class_name: "encoder"`): one 3D Vision Transformer (ViT) per temporal phase, with **weights shared across phases**, followed by an **LSTM fusion module** producing the latent representation.
The two hparams files are referenced together in the [experiment configuration](./ViT-LSTM_EXPERIMENT_CONFIGURATION.md#composite-model).
 
---


## ViT-LSTM Encoder Configuration
 
The encoder uses the generic **multimodal encoder**: each declared modality gets its own encoder, and a fusion module combines the resulting embeddings. Here the modalities are the temporal phases produced by the [pipeline](./ViT-LSTM_PIPELINE_CONFIGURATION.md) (`phase_0`, `phase_2`, ...).
 
One hparams file exists **per phase configuration**, all sharing the same architecture and differing only by the set of declared phase modalities:
 
| File | Phases (modalities) | Used with pipeline |
|------|---------------------|--------------------|
| `vit-lstm_hparams.json` | `phase_0` ... `phase_12` (13) | `4d_pipeline.json` |
| `vit-lstm_hparams_7phases.json` | 0, 2, 4, 6, 8, 10, 12 | `4d_pipeline_7phases.json` |
| `vit-lstm_hparams_4phases.json` | 0, 4, 8, 12 | `4d_pipeline_4phases.json` |
| `vit-lstm_hparams_3phases.json` | 0, 6, 12 | `4d_pipeline_3phases.json` |
| `vit-lstm_hparams_2phases_mid.json` | 4, 8 | `4d_pipeline_2phases_mid.json` |
| `vit-lstm_hparams_2phases_firstlast.json` | 0, 12 | `4d_pipeline_2phases_firstlast.json` |
 
Example (`vit-lstm_hparams_3phases.json`):
 
```json
{
    "encoder_config": {
        "architecture": "multimodal",
        "encoders_configs": {
            "phase_0": {
                "architecture": "vit",
                "in_channels": 1,
                "spatial_dims": 3,
                "patch_size": [16, 16, 16],
                "num_vit_layers": 8,
                "num_vit_heads": 6,
                "hidden_size": 192,
                "mlp_dim": 384,
                "output_dimension": 192,
                "proj_type": "perceptron",
                "image_size": [48, 48, 48],
                "pooling": "cls"
            },
            "phase_6": "phase_0",
            "phase_12": "phase_0"
        },
        "fusion_module_config": {
            "fusion_mode": "lstm",
            "input_size": 192,
            "hidden_size": 128,
            "learn_initial_state": true,
            "pooling": "last",
            "num_layers": 4,
            "bias": true,
            "dropout": 0.0,
            "bidirectional": false,
            "proj_size": 0
        },
        "output_dimension": 128
    },
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 1e-4
        }
    }
}
```
 
### Per-phase ViT encoders (`encoders_configs`)
 
One entry per phase modality. The **first** phase defines the full ViT configuration; the other phases simply **reference it by name** (e.g. `"phase_12": "phase_0"`), which means they **share the same encoder weights**. The keys must match the phase modality names output by the pipeline.
 
| Parameter | Description |
|------------|-------------|
| `architecture` | Per-phase backbone (`vit`, a 3D Vision Transformer). |
| `in_channels` | Number of image channels (1, grayscale volumes). |
| `spatial_dims` | Number of spatial dimensions of each volume (3). |
| `patch_size` | Size of the 3D patches tokenized by the ViT (`16 × 16 × 16`). |
| `num_vit_layers` | Number of transformer layers (8). |
| `num_vit_heads` | Number of attention heads (6). |
| `hidden_size` | Dimension of the token embeddings (192). |
| `mlp_dim` | Dimension of the transformer feed-forward layers (384). |
| `output_dimension` | Size of the per-phase embedding (192). |
| `proj_type` | Patch-embedding projection type (`perceptron`). |
| `image_size` | Spatial size of each input volume (`48 × 48 × 48`, matching the `resize` step of the pipeline). |
| `pooling` | Token pooling producing the phase embedding (`cls`: the class token is used). |
 
### LSTM fusion module (`fusion_module_config`)
 
The phase embeddings, ordered temporally, form the input sequence of an LSTM:
 
| Parameter | Description |
|------------|-------------|
| `fusion_mode` | Type of fusion module (`lstm`). |
| `input_size` | Dimension of each sequence element; must equal the per-phase `output_dimension` (192). |
| `hidden_size` | Dimension of the LSTM hidden state (128). |
| `learn_initial_state` | Whether the initial hidden/cell states are learned parameters. |
| `pooling` | How the sequence of hidden states is reduced (`last`: the last hidden state is kept). |
| `num_layers` | Number of stacked LSTM layers (4). |
| `bias` | Whether bias terms are used. |
| `dropout` | Dropout between LSTM layers. |
| `bidirectional` | Whether the LSTM is bidirectional. |
| `proj_size` | Optional output projection size of the LSTM (0: disabled). |
 
### Encoder output
 
| Parameter | Description |
|------------|-------------|
| `output_dimension` | Size of the final representation passed to the classifier (128, the LSTM hidden size). |
 
> [!CAUTION]
> The phase modality names in `encoders_configs` must exactly match the outputs of the [pipeline](./ViT-LSTM_PIPELINE_CONFIGURATION.md), and `input_size` of the fusion module must equal the per-phase `output_dimension`. When changing the number of phases of an experiment, the encoder hparams **and** the pipeline must be changed together.
 
---
 
## Optimizer Configuration
 
Identical to the [Ultra example](../Ultra/Ultra_HPARAMS_CONFIGURATION.md#optimizer-configuration): both the classifier and the encoder are optimized with Adam and a learning rate of `1e-4`.
 
---
 
## Classifier (MLP) Configuration
 
Identical to the [Ultra example](../Ultra/Ultra_HPARAMS_CONFIGURATION.md#classifier-mlp-configuration): two variants exist, `mlp_2class-hparams.json` (B-M and BG-M tasks) and `mlp_3class-hparams.json` (B-G-M task), differing only by `class_count`.
 
---
 
## Configuration Summary
 
This configuration defines a 4D image classification model with:
 
- One 3D ViT per temporal phase (8 layers, 6 heads, hidden size 192, patches 16³ on 48³ volumes, CLS pooling), **weights shared across phases**
- 192-dimensional per-phase embeddings
- Sequential fusion by a 4-layer unidirectional LSTM (hidden size 128, learned initial state, last hidden state kept)
- 128-dimensional final representation
- A linear MLP head with 2 or 3 output classes depending on the classification task
- End-to-end training (encoder not frozen)
- Adam optimization with a learning rate of `1e-4`

 
