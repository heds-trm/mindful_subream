# Model HParams Configuration
 
This configuration defines the composite classification model used in the [4DUF example](./4DUF_README.md). The model is made of two parts, each configured by its own hparams file:

- an **MLP classifier** (top-level model, `class_name: "mlp"`) for the final prediction,
- an **"Ultra" 4D encoder** (sub-module `representation_model`, `class_name: "encoder"`) transforming the 4D image into a latent representation.
The two hparams files are referenced together in the [experiment configuration](./Ultra_EXPERIMENT_CONFIGURATION.md#composite-model).
 
---

## Ultra Encoder Configuration
 
The encoder is a transformer ("Ultra" architecture) that processes a sequence of 3D volumes, one per temporal phase. One hparams file exists **per phase count**, all sharing the same architecture and differing only by `phase_count`:
 
| File | `phase_count` | Used with pipeline |
|------|---------------|--------------------|
| `ultra_hparams.json` | 13 | `ultra-standardize.json` (full sequence) |
| `ultra_hparams_7phases.json` | 7 | `..._7slctPhase.json` |
| `ultra_hparams_4phases.json` | 4 | `..._4slctPhase.json` |
| `ultra_hparams_3phases.json` | 3 | `..._3slctPhase.json` |
| `ultra_hparams_2phases.json` | 2 | `..._2slctPhase_mid.json` or `..._2slctPhase_firstlast.json` |

 
Example (`ultra_hparams.json`):
 
```json
{
    "encoder_config": {
        "architecture": "ultra",
        "in_channels": 1,
        "image_size": [8, 8, 8],
        "phase_count": 13,
        "hidden_size": 384,
        "mlp_dim": 1536,
        "num_layers": 4,
        "num_heads": 6,
        "output_dimension": 128,
        "use_class_token": false,
        "dropout_rate": 0.0,
        "spatial_dims": 3,
        "qkv_bias": false,
        "pooling": "max",
        "norm_outputs": false
    },
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 1e-4
        }
    }
}
```
 
| Parameter | Description |
|------------|-------------|
| `architecture` | Encoder backbone (`ultra`, the 4D transformer encoder). |
| `in_channels` | Number of image channels (1, grayscale volumes). |
| `image_size` | Spatial size of each 3D volume after resizing (`8 × 8 × 8`, matching the `resize` step of the pipeline). |
| `phase_count` | Number of temporal phases in the input sequence. **Must match** the number of phases produced by the pipeline (`filter_slices`). |
| `hidden_size` | Dimension of the transformer token embeddings. |
| `mlp_dim` | Dimension of the transformer feed-forward layers. |
| `num_layers` | Number of transformer layers. |
| `num_heads` | Number of attention heads. |
| `output_dimension` | Size of the latent representation produced by the encoder (128). |
| `use_class_token` | Whether a class token is prepended to the token sequence. Here, disabled: the representation is obtained by pooling. |
| `dropout_rate` | Dropout probability. |
| `spatial_dims` | Number of spatial dimensions of each volume (3). |
| `qkv_bias` | Whether biases are used in the query/key/value projections. |
| `pooling` | Pooling operation applied over the tokens to produce the final representation (`max`). |
| `norm_outputs` | Whether the output representation is normalized. |
 
> [!CAUTION]
> When changing the number of phases of an experiment, `phase_count` in the encoder hparams and `filter_slices` in the pipeline must stay consistent, otherwise the transformer input shape will not match.
 
---
 
## Optimizer Configuration
 
Both the classifier and the encoder use the same optimizer configuration:
 
```json
{
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 1e-4
        }
    }
}
```
 
| Parameter | Description |
|------------|-------------|
| `optimizer_type` | Optimization algorithm used during training. |
| `lr` | Learning rate. |
 
> [!NOTE]
> - The Adam optimizer is used for gradient-based optimization.
> - The learning rate is set to `1e-4`.
 
---




## Classifier (MLP) Configuration
 
Two variants exist, differing only by the number of output classes:
 
- `mlp_2class-hparams.json` — binary classification (B-M and BG-M tasks), `class_count = 2`,
- `mlp_3class-hparams.json` — 3-class classification (B-G-M task), `class_count = 3`.
Example (`mlp_3class-hparams.json`):

```json
{
    "classifier_config": {
        "features": [],
        "class_count": 3,
        "yield_confidence": false
    },
    "freeze_representation_model": false,
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 1e-4
        }
    }
}
```
 
| Parameter | Description |
|------------|-------------|
| `features` | Hidden layer sizes of the MLP head. An empty list means a single linear layer on top of the encoder representation. |
| `class_count` | Number of output classes (2 or 3 depending on the class grouping). |
| `yield_confidence` | Whether the model returns confidence values together with predictions. |
| `freeze_representation_model` | Whether the encoder sub-module weights are frozen during training. Set to `false`, so the encoder is trained end-to-end with the classifier. |
 
---


## Configuration Summary
 
This configuration defines a 4D image classification model with:
 
- A transformer "Ultra" encoder over the temporal phases (4 layers, 6 heads, hidden size 384)
- 3D volumes resized to `8 × 8 × 8`, single channel
- Max pooling over tokens, 128-dimensional representation
- A linear MLP head with 2 or 3 output classes depending on the class grouping
- End-to-end training (encoder not frozen)
- Adam optimization with a learning rate of `1e-4`
Variants of the encoder hparams only change `phase_count` (13, 10, 7, 4, 3 or 2 phases) to study the impact of temporal subsampling on classification performance.
