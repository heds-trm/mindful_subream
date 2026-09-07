# Multimodal SIEVE Model Configuration

This configuration defines a multimodal SIEVE-based classification model[^1] that combines image features and categorical features. It specifies the encoders, SIEVE aggregation settings, classifier settings, and optimizer configuration.

## Overview

The model is configured for binary classification and can optionally output confidence scores. It uses:

- a DenseNet121 encoder for images,
- a scalar encoder for categorical features,
- a SIEVE module to fuse modalities,
- a classifier head for final prediction.

---

## Main Parameters

| Parameter | Description |
|------------|-------------|
| `class_count` | Number of output classes. |
| `output_confidence` | Whether the model returns confidence values together with predictions. |
| `sieve_mutual_lambda` | Weight applied to the mutual component of the SIEVE objective. |
| `sieve_exclusive_lambda` | Weight applied to the exclusive component of the SIEVE objective. |

```json
{
    "class_count": 2,
    "output_confidence": false,
    "sieve_mutual_lambda": 0.4,
    "sieve_exclusive_lambda": 0.4
}
```

---

## Optimizer Configuration

The optimizer defines how model parameters are updated during training. This section is similar to the [unimodal version](https://github.com/heds-trm/mindful_core/blob/main/examples/mura/MURA_HPARAMS_CONFIGURATION.md#optimizer-configuration).

```json
{
    "optimizer_config": {
        "optimizer": {
            "optimizer_type": "adam",
            "lr": 0.0001
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

## Encoders Configuration

The `encoders_config` section defines how each modality is encoded before fusion.

```json
{
    "encoders_config": {
        "image": {
            "architecture": "densenet121",
            "spatial_dims": 2,
            "in_channels": 1,
            "output_dimension": 128
        },
        "categorical_features": {
            "architecture": "scalar",
            "input_size": 8,
            "hidden_sizes": [],
            "output_dimension": 128
        }
    }
}
```

### Image Encoder

| Parameter | Description |
|------------|-------------|
| `architecture` | Backbone used for image encoding. |
| `spatial_dims` | Number of spatial dimensions in the input data. |
| `in_channels` | Number of image channels. |
| `output_dimension` | Size of the latent representation produced by the encoder. |

### Categorical Features Encoder

| Parameter | Description |
|------------|-------------|
| `architecture` | Encoder architecture used for categorical features. |
| `input_size` | Number of categorical input features. |
| `hidden_sizes` | Hidden layer dimensions of the encoder. An empty list indicates no hidden layers. |
| `output_dimension` | Size of the latent representation produced by the encoder. |

> [!NOTE]
> The image encoder uses a DenseNet121 backbone to extract visual features.  
> The categorical encoder transforms structured metadata into a latent representation.  
> Both encoders produce embeddings of dimension `128`, enabling multimodal fusion.

---

## SIEVE Configuration

The `sieve_config` section controls the multimodal fusion module.

```json
{
    "sieve_config": {
        "head_count": 4,
        "layer_count": 4,
        "add_position_embeddings": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `head_count` | Number of attention heads used by the SIEVE module. |
| `layer_count` | Number of layers in the SIEVE module. |
| `add_position_embeddings` | Whether positional embeddings are added to the modality representations. |

---

## Classifier Configuration

The classifier processes the fused representation and produces the final prediction.

```json
{
    "classifier_config": {
        "head_count": 4,
        "layer_count": 4,
        "add_position_embeddings": true,
        "pre_norm_outputs": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `head_count` | Number of attention heads used by the classifier. |
| `layer_count` | Number of classifier layers. |
| `add_position_embeddings` | Whether positional embeddings are used. |
| `pre_norm_outputs` | Whether outputs are normalized before the final prediction layer. |

---

## Configuration Summary

This configuration defines a multimodal binary classification model with:

- Image encoding using a DenseNet121 model
- Structured categorical feature encoding
- 128-dimensional modality embeddings
- SIEVE-based multimodal fusion
- 4 fusion layers with 4 attention heads
- 4 classifier layers with 4 attention heads
- Adam optimization with a learning rate of 0.0001

The model is designed to jointly leverage imaging data and structured metadata for classification tasks.

[^1]: Lokaj, B., Durand de Gevigney, V., Djema, D. A., Zaghir, J., Goldman, J. P., Bjelogrlic, M., Turbe, H., Kinkel, K., Lovis, C., Schmid, J. (2025). Multimodal deep learning fusion of ultrafast-DCE MRI and clinical information for breast lesion classification. Computers in biology and medicine. [10.1016/j.compbiomed.2025.109721](https://doi.org/10.1016/j.compbiomed.2025.109721)
