# Pipeline Configuration

This pipeline defines the preprocessing and augmentation operations applied to the 4D imaging data before training and evaluation.

The pipeline processes two modalities:

- `image`: 4D medical image data (3D volume × temporal phases)
- `label`: ground-truth classification label

Several pipeline files exist; they are all identical except for the **phase selection** performed at loading time (`filter_slices` parameter):

| File | `filter_slices` | Phases kept |
|------|-----------------|-------------|
| `ultra-standardize.json` | (all phases) | all 13 phases |
| `ultra-standardize_7slctPhase.json` | `[0, 2, 4, 6, 8, 10, 12]` | 7 phases |
| `ultra-standardize_4slctPhase.json` | `[0, 4, 8, 12]` | 4 phases |
| `ultra-standardize_3slctPhase.json` | `[0, 6, 12]` | first, middle, last |
| `ultra-standardize_2slctPhase_mid.json` | `[4, 8]` | 2 middle phases |
| `ultra-standardize_2slctPhase_firstlast.json` | `[0, 12]` | first and last phases |

> [!CAUTION]
> The number of selected phases must match the `phase_count` of the encoder hparams used by the experiment (see [HParams configuration](./Ultra_HPARAMS_CONFIGURATION.md#ultra-encoder-configuration)).

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
> [!NOTE]
> Prepare your 4D data according to guidelines in the Preprocessing stage.
> 
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
        "image_only": true,
        "filename_pattern":"phase_*.mha"
    }
}
```

| Parameter | Description |
|------------|-------------|
| `filter_slices` | Indices of the temporal phases to keep. When omitted, the full sequence is loaded. |
| `image_only` | Whether only the image data is returned (metadata discarded). |
| `filename_pattern` | Filename pattern to identify 3D volumes of phases to load. Optional; defaults to `phase_*.mha`. Change it to match your own file naming, e.g. `"t_*.nii.gz"`. |

> [!CAUTION]
> `load_image_4d` loads 4D image data as a series of 3D volumes saved to files following a specific filename pattern. By default (i.e. if `filename_pattern` is missing in parameters), this pattern is `phase_*.mha`.
>
> 
> The image filename specified in the CSV file as the `image:image` entry (see [datasets section](../4DUF_README.md#tasks-and-datasets)) will provide the folder containing these 3D volumes. Then the loader will load all phases in the folder matching the `filename pattern`, in phase order.
>
> For example, let's assume we have 3 phases and the CSV file has this first row:
> ```csv
> ScanID,SubsetID,Label,image:image,image:mask
> 66_1_0,test,0,image_folder/phase_2.mha,mask_folder/mask_0.mha
> ```
> The folder `image_folder` will contain 3 files named `phase_0.mha`, `phase_1.mha`, `phase_2.mha`
> **Accepted image formats are those supported by MONAI `LoadImage`.**

### Standardize Intensity

Standardizes voxel intensities over the whole volume.

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
| `range_x` / `range_y` / `range_z` | Rotation ranges (radians) around each axis. |
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
5. Apply image augmentations during training: Random flipping, Random rotations, Random translations (up to 2 voxels per axis), Gaussian noise addition

