# Model HParams Configuration
 
This configuration defines the composite classification model used in the [4DUF example](../4DUF_README.md). The model is made of two parts, each configured by its own hparams file:

- an **MLP classifier** (classifier model, `class_name: "mlp"`) for the final prediction,
- a **ViT-Transformer encoder** (sub-module `representation_model`, `class_name: "encoder"`): one 3D Vision Transformer (ViT) per temporal phase, with **weights shared across phases**, followed by an **Transformer fusion module** producing the latent representation.
The two hparams files are referenced together in the [experiment configuration](./ViT-Transformer_EXPERIMENT_CONFIGURATION.md#composite-model).
 
---


## ViT-Transformer Encoder Configuration
 
The encoder uses the generic **multimodal encoder**, as in the [ViT-LSTM example](../ViT-LSTM/ViT-LSTM_HPARAMS_CONFIGURATION.md#vit-lstm-encoder-configuration): each declared modality gets its own encoder, and a fusion module combines the resulting embeddings. Here the modalities are the temporal phases produced by the [pipeline](./ViT-Transformer_PIPELINE_CONFIGURATION.md) (`phase_0`, `phase_2`, ...).
 
Compared to the ViT-LSTM model, two things change:
 
- the per-phase ViT is **lighter** (4 transformer layers instead of 8) and uses **max pooling** over the tokens instead of the class token,
- the phase embeddings are fused by a **Transformer** instead of an LSTM.
 
One hparams file exists **per phase configuration**, all sharing the same architecture and differing only by the set of declared phase modalities:
 
| File | Phases (modalities) | Used with pipeline |
|------|---------------------|--------------------|
| `vit-transformer_hparams.json` | `phase_0` ... `phase_12` (13) | `4d_pipeline.json` |
| `vit-transformer_hparams_7phases.json` | 0, 2, 4, 6, 8, 10, 12 | `4d_pipeline_7phases.json` |
| `vit-transformer_hparams_4phases.json` | 0, 4, 8, 12 | `4d_pipeline_4phases.json` |
| `vit-transformer_hparams_3phases.json` | 0, 6, 12 | `4d_pipeline_3phases.json` |
| `vit-transformer_hparams_2phases_mid.json` | 4, 8 | `4d_pipeline_2phases_mid.json` |
| `vit-transformer_hparams_2phases_firstlast.json` | 0, 12 | `4d_pipeline_2phases_firstlast.json` |
 
Example (`vit-transformer_hparams_3phases.json`):
 

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
                "num_vit_layers": 4,
                "num_vit_heads": 6,
                "hidden_size": 192,
                "mlp_dim": 384,
                "output_dimension": 192,
                "proj_type": "perceptron",
                "image_size": [48, 48, 48],
                "pooling": "max"
            },
            "phase_6": "phase_0",
            "phase_12": "phase_0"
        },
        "fusion_module_config": {
            "fusion_mode": "transformer",
            "input_size": 192,
            "head_count": 6,
            "layer_count": 2,
            "add_cls_token": false,
            "add_position_embeddings": true,
            "modality_count": 13,
            "pooling": "auto",
            "project_output": true,
            "output_dimension": 128,
            "pre_norm_outputs": false,
            "pre_activation": "ReLU",
            "output_activation": null
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
 
As in the [ViT-LSTM example](../ViT-LSTM/ViT-LSTM_HPARAMS_CONFIGURATION.md#per-phase-vit-encoders-encoders_configs): one entry per phase modality, the **first** phase defines the full ViT configuration and the other phases **reference it by name** (e.g. `"phase_12": "phase_0"`), which means they **share the same encoder weights**. The keys must match the phase modality names output by the pipeline.
 
The parameters are the same as in the ViT-LSTM example, with two differences:
 
| Parameter | Description |
|-----------|-------------|
| `num_vit_layers` | Number of transformer layers (**4** instead of 8 in the ViT-LSTM example). |
| `pooling` | Token pooling producing the phase embedding (**`max`** pooling over the tokens instead of the class token). |

 
### Transformer fusion module (`fusion_module_config`)
 
The phase embeddings form the token sequence of a Transformer:
 
| Parameter | Description |
|-----------|-------------|
| `fusion_mode` | Type of fusion module (`transformer`). |
| `input_size` | Dimension of each token; must equal the per-phase `output_dimension` (192). |
| `head_count` | Number of attention heads (6). |
| `layer_count` | Number of transformer layers (2). |
| `add_cls_token` | Whether a class token is prepended to the token sequence. Here, disabled: the representation is obtained by pooling. |
| `add_position_embeddings` | Whether positional embeddings are added to the phase tokens, so that the fusion is aware of the temporal order of the phases. |
| `modality_count` | Maximum number of modality positions declared for the positional embeddings. Kept at 13 (the full sequence) in all phase variants. |
| `pooling` | Pooling operation applied over the output tokens to produce the fused representation (`auto`). |
| `project_output` | Whether the pooled representation is projected to `output_dimension` by a final linear layer. |
| `output_dimension` | Size of the fused representation (128). |
| `pre_norm_outputs` | Whether outputs are normalized before the projection layer. |
| `pre_activation` | Activation applied before the output projection (`ReLU`). |
| `output_activation` | Activation applied to the projected output (`null`: none). |

 
### Encoder output
 
| Parameter | Description |
|-----------|-------------|
| `output_dimension` | Size of the final representation passed to the classifier (128). |
 
> [!CAUTION]
> The phase modality names in `encoders_configs` must exactly match the outputs of the [pipeline](./ViT-Transformer_PIPELINE_CONFIGURATION.md), and `input_size` of the fusion module must equal the per-phase `output_dimension`. When changing the number of phases of an experiment, the encoder hparams **and** the pipeline must be changed together.
 
---

 
## Optimizer Configuration
 
Identical to the [Ultra example](../Ultra/Ultra_HPARAMS_CONFIGURATION.md#optimizer-configuration): both the classifier and the encoder are optimized with Adam and a learning rate of `1e-4`.
 
---
 
## Classifier (MLP) Configuration
 
Identical to the [Ultra example](../Ultra/Ultra_HPARAMS_CONFIGURATION.md#classifier-mlp-configuration): two variants exist, `mlp_2class-hparams.json` (B-M and BG-M tasks) and `mlp_3class-hparams.json` (B-G-M task), differing only by `class_count`.
 
---
 
## Configuration Summary
 
This configuration defines a 4D image classification model with:
 
- One 3D ViT per temporal phase (4 layers, 6 heads, hidden size 192, patches 16³ on 48³ volumes, max pooling), **weights shared across phases**
- 192-dimensional per-phase embeddings
- Fusion by a 2-layer Transformer (6 heads, positional embeddings over the phase sequence, no class token)
- 128-dimensional final representation
- A linear MLP head with 2 or 3 output classes depending on the classification task
- End-to-end training (encoder not frozen)
- Adam optimization with a learning rate of `1e-4`

 
