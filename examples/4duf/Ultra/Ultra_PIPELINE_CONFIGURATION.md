# Pipeline Configuration

This pipeline defines the preprocessing and augmentation operations applied to the 4D imaging data before training and evaluation.

The pipeline processes two modalities:

- `image`: 4D medical image data (3D volume × temporal phases)
- `label`: ground-truth classification label

Several pipeline files exist; they are all identical except for the **phase selection** performed at loading time (`filter_slices` parameter):

| File | `filter_slices` | Phases kept |
|------|-----------------|-------------|
| `ultra-standardize.json` | — | all 13 phases |
| `ultra-standardize_7slctPhase.json` | `[0, 2, 4, 6, 8, 10, 12]` | 7 phases |
| `ultra-standardize_4slctPhase.json` | `[0, 4, 8, 12]` | 4 phases |
| `ultra-standardize_3slctPhase.json` | `[0, 6, 12]` | first, middle, last |
| `ultra-standardize_2slctPhase_mid.json` | `[4, 8]` | 2 middle phases |
| `ultra-standardize_2slctPhase_firstlast.json` | `[0, 12]` | first and last phases |
| `ultra-standardize_slctPhase.json` | `[2, 10]` | 2 intermediate phases |

> [!CAUTION]
> The number of selected phases must match the `phase_count` of the encoder hparams used by the experiment (see [HParams configuration](./4DUF_HPARAMS_CONFIGURATION.md#ultra-encoder-configuration)).

---

# Inputs and Outputs

## Inputs

| Name | Description |
|--------|-------------|
| `image` | Input 4D medical image. |
| `label` | Ground-truth label. |

```json
{
    "inputs": {
        "image": "image",
        "label": "label"
    }
}
```

## Outputs

| Name | Description |
|--------|-------------|
| `image` | Processed image (sequence of standardized, resized 3D volumes). |
| `label` | Processed label. |

```json
{
    "outputs": {
        "image": "image",
        "label": "label"
    }
}
```

---

# Preprocessing Stage

The `preprocess` stage prepares both modalities for model consumption.

## Image Processing

### Load 4D Image (with phase selection)

```json
{
    "type": "load_image_4d",
    "parameters": {
        "filter_slices": [0, 2, 4, 6, 8, 10, 12],
        "image_only": true
    }
}
```

| Parameter | Description |
|------------|-------------|
| `filter_slices` | Indices of the temporal phases to keep. When omitted, the full sequence is loaded. |
| `image_only` | Whether only the image data is returned (metadata discarded). |

### Standardize Intensity

Standardizes voxel intensities (zero mean, unit variance) over the whole volume.

```json
{
    "type": "standardize_intensity",
    "parameters": {
        "channel_wise": false,
        "spatial_dims": 3
    }
}
```

| Parameter | Description |
|------------|-------------|
| `channel_wise` | Whether standardization is computed per channel. Here computed globally. |
| `spatial_dims` | Number of spatial dimensions of each volume. |

### Resize

Resizes each 3D volume to a fixed resolution matching the encoder's expected `image_size`.

```json
{
    "type": "resize",
    "parameters": {
        "spatial_size": [8, 8, 8]
    }
}
```

### Ensure Type

Converts the image into a tensor with the desired type and device.

```json
{
    "type": "ensure_type",
    "parameters": {
        "device": "cpu",
        "dtype": "float32"
    }
}
```

## Label Processing

Converts labels into tensors.

```json
{
    "type": "to_tensor",
    "parameters": {
        "dtype": "float32",
        "device": "cpu"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `dtype` | Tensor data type. |
| `device` | Target device. |

---

# View Augmentation Stage

The `view_augment` stage applies image augmentations during training only.

## Random Flip

```json
{
    "type": "rand_flip",
    "parameters": {
        "prob": 0.5
    }
}
```

## Random Rotation

```json
{
    "type": "rand_rotate",
    "parameters": {
        "prob": 0.8,
        "range_x": 0.34,
        "range_y": 0.34,
        "range_z": 0.34,
        "padding_mode": "zeros"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the rotation. |
| `range_x` / `range_y` / `range_z` | Rotation ranges (radians) around each axis, roughly ±19.5°. |
| `padding_mode` | Padding strategy for voxels leaving the field of view. |

## Random Affine (translation)

```json
{
    "type": "rand_affine",
    "parameters": {
        "prob": 0.5,
        "translate_range": [2, 2, 2],
        "padding_mode": "border"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the transform. |
| `translate_range` | Maximum translation (voxels) along each axis. |
| `padding_mode` | Padding strategy at borders. |

## Random Gaussian Noise

```json
{
    "type": "rand_gaussian_noise",
    "parameters": {
        "prob": 0.2,
        "std": 0.01
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of adding noise. |
| `std` | Standard deviation of the Gaussian noise. |

> [!NOTE]
> Augmentations are applied consistently to the whole 4D sequence, so all phases of a sample undergo the same geometric transformation.

---

# Pipeline Summary

This pipeline performs the following operations:

1. Load the 4D image, optionally selecting a subset of temporal phases (`filter_slices`).
2. Standardize voxel intensities.
3. Resize all 3D volumes to **8 × 8 × 8**.
4. Convert images and labels to `float32` tensors.
5. Apply image augmentations during training:
   - Random flipping
   - Random rotations (up to ~±19.5° per axis)
   - Random translations (up to 2 voxels per axis)
   - Gaussian noise addition

The phase-selection variants allow studying the trade-off between the temporal richness of the 4D sequence and the model complexity, at constant preprocessing and augmentation.