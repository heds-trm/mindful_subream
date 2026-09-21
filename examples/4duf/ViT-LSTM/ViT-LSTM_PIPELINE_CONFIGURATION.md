# Pipeline Configuration

This pipeline defines the preprocessing and augmentation operations applied to the 4D imaging data before training and evaluation.
 
Its distinctive feature, compared to the [Ultra pipeline](../Ultra/Ultra_PIPELINE_CONFIGURATION.md), is that the 4D image is **split into one modality per temporal phase** (`phase_0`, `phase_2`, ...): the ViT-LSTM encoder expects each phase as a separate input.
 
One pipeline file exists per phase configuration:
 
| File | `filter_slices` | Output modalities |
|------|-----------------|-------------------|
| `4d_pipeline.json` | (all phases) | `phase_0` ... `phase_12` |
| `4d_pipeline_7phases.json` | `[0, 2, 4, 6, 8, 10, 12]` | `phase_0`, `phase_2`, ..., `phase_12` |
| `4d_pipeline_4phases.json` | `[0, 4, 8, 12]` | `phase_0`, `phase_4`, `phase_8`, `phase_12` |
| `4d_pipeline_3phases.json` | `[0, 6, 12]` | `phase_0`, `phase_6`, `phase_12` |
| `4d_pipeline_2phases_mid.json` | `[4, 8]` | `phase_4`, `phase_8` |
| `4d_pipeline_2phases_firstlast.json` | `[0, 12]` | `phase_0`, `phase_12` |
 
> [!CAUTION]
> The output modality names must exactly match the `encoders_configs` entries of the [encoder hparams](./ViT-LSTM_HPARAMS_CONFIGURATION.md#vit-lstm-encoder-configuration) used by the experiment.
 
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
 
Each kept phase becomes its own output modality, mapped from the `image` input. Example for the 3 phases pipeline:
 
```json
{
    "outputs": {
        "phase_0": "image",
        "phase_6": "image",
        "phase_12": "image",
        "label": "label"
    }
}
```
 
---
 
# Preprocessing Stage
 
The `preprocess` stage prepares all modalities for model consumption.
 
## Image Processing
 
### Load 4D Image (with phase selection)
 
```json
{
    "type": "load_image_4d",
    "parameters": {
        "image_only": true,
        "filter_slices": [0, 6, 12],
        "separate_channel_dim": false,
        "filename_pattern":"phase_*.mha"
    }
}
```
 
| Parameter | Description |
|------------|-------------|
| `filter_slices` | Indices of the temporal phases to keep. Omitted in the 13-phase pipeline (full sequence). |
| `image_only` | Whether only the image data is returned (metadata discarded). |
| `separate_channel_dim` | Whether a separate channel dimension is created at loading; disabled, the phases are kept along the first axis so they can be split by `take_slice`. |
| `filename_pattern` | Filename pattern to identify 3D volumes of phases to load. Optional; defaults to `phase_*.mha`. Change it to match your own file naming, e.g. `"t_*.nii.gz"`. |

> [!CAUTION]
> `load_image_4d` loads 4D image data as a series of 3D volumes saved to files following a specific filename pattern. By default (i.e. if `filename_pattern` is missing in parameters), this pattern is `phase_*.mha`.
> 
> The image filename specified in the CSV file as the `image:image` entry (see [datasets section](../4DUF_README.md#tasks-and-datasets)) will provide the folder containing these 3D volumes. Then the loader will load all phases in the folder matching the `filename pattern`, in phase order.
>
> For example, let's assume we have 3 phases and the CSV file has this first row:
> ```csv
> ScanID,SubsetID,Label,image:image,image:mask
> 66_1_0,test,0,image_folder/phase_3.mha,mask_folder/mask_0.mha
> ```
> The folder `image_folder` will contain 3 files named `phase_0.mha`, `phase_1.mha`, `phase_2.mha`
> **Accepted image formats are those supported by MONAI `LoadImage`.**
 
### Scale Intensity
 
```json
{
    "type": "scale_intensity",
    "parameters": {
        "channel_wise": false
    }
}
```

### Resize
 
Resizes each volume to the ViT input resolution:
 
```json
{
    "type": "resize",
    "parameters": {
        "spatial_size": [48, 48, 48]
    }
}
```
 
### Split into phase modalities (`take_slice`)
 
One `take_slice` operation per kept phase extracts the corresponding 3D volume along the temporal axis and writes it to a dedicated output modality:
 
```json
{
    "modalities": ["image"],
    "type": "take_slice",
    "parameters": {"axis": 0, "slice_index": 1},
    "outputs": ["phase_6"]
}
```
 
| Parameter | Description |
|------------|-------------|
| `axis` | Axis along which the slice is taken (0: temporal/phase axis). |
| `slice_index` | Index **within the loaded (filtered) sequence**. |
| `outputs` | Name of the created phase modality. |
 
> [!CAUTION]
> `slice_index` refers to the position **after** `filter_slices` has been applied, not to the original phase number. For example, in the 3-phase pipeline, `slice_index: 1` corresponds to original phase 6 and is output as `phase_6`.
 
### Ensure channel and type
 
The per-phase volumes are given a channel-first layout and converted to `float32` tensors. These operations (and the augmentations below) use the dictionary-based variants (`..._d` suffix) so that they apply to several modalities at once:
 
```json
{
    "modalities": ["phase_0", "phase_6", "phase_12"],
    "type": "ensure_channel_firstd",
    "parameters": {}
},
{
    "modalities": ["phase_0", "phase_6", "phase_12"],
    "type": "ensure_typed",
    "parameters": {
        "device": "cpu",
        "dtype": "float32"
    }
}
```
 
## Label Processing
 
Identical to the [Ultra pipeline](../Ultra/Ultra_PIPELINE_CONFIGURATION.md#label-processing): labels are converted to `float32` tensors with `to_tensor`.
 
---
 
# View Augmentation Stage
 
The augmentations and their parameters are identical to the [Ultra pipeline](../Ultra/Ultra_PIPELINE_CONFIGURATION.md#view-augmentation-stage), except that the dictionary-based variants are used and applied **jointly to all phase modalities**, so every phase of a sample undergoes the **same** random transformation:
 
- `rand_flipd` — random flip, `prob = 0.5`;
- `rand_rotated` — random rotation, `prob = 0.8`, ranges `±0.34` rad per axis, zero padding;
- `rand_affined` — random translation, `prob = 0.5`, up to 2 voxels per axis, border padding;
- `rand_gaussian_noised` — Gaussian noise, `prob = 0.2`, `std = 0.01`.
---
 
# Pipeline Summary
 
This pipeline performs the following operations:
 
1. Load the 4D image, optionally selecting a subset of temporal phases (`filter_slices`).
2. Rescale voxel intensities.
3. Resize all 3D volumes to **48 × 48 × 48**.
4. Split the sequence into one modality per phase (`take_slice`), named `phase_<original_index>`.
5. Ensure channel-first layout and convert phases and labels to `float32` tensors.
6. Apply consistent image augmentations to all phases during training: random flipping, random rotations, random translations, Gaussian noise addition.
