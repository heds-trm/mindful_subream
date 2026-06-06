# Multimodal Pipeline Configuration

This pipeline defines the preprocessing and augmentation operations applied to both imaging and categorical data before training and evaluation.

The pipeline processes three modalities:

- `image`: medical image data
- `categorical`: structured metadata
- `label`: ground-truth classification label

---

# Inputs and Outputs

## Inputs

| Name | Description |
|--------|-------------|
| `image` | Input medical image. |
| `categorical` | Structured categorical metadata associated with the sample. |
| `label` | Ground-truth label. |

```json
{
    "inputs": {
        "image": "image",
        "categorical": "categorical",
        "label": "label"
    }
}
```

## Outputs

| Name | Description |
|--------|-------------|
| `image` | Processed image. |
| `categorical` | Processed categorical features. |
| `label` | Processed label. |

```json
{
    "outputs": {
        "image": "image",
        "categorical": "categorical",
        "label": "label"
    }
}
```

---

# Preprocessing Stage

The `preprocess` stage prepares all modalities for model consumption.

## Image Processing

### Load Image

The image pipeline is identical to the pipeline in [unimodal mode](https://github.com/heds-trm/mindful_core/blob/main/examples/mura/MURA_PIPELINE_CONFIGURATION.md#processing-sequence):
- Loads the image from disk.
- Converts images to grayscale.
- Resizes images to a fixed resolution.
- Normalizes image intensities.
- Converts the image into a tensor with the desired type.

---

## Categorical Feature Processing

The fusion of other type of data to image data is at the core of the multimodal fusion. Here, we include categorical data corresponding to body part acquired in the X-ray image.
The categorical preprocessing stage converts structured metadata into a machine-learning-friendly representation.

```json
{
    "type": "categorical_preprocess",
    "parameters": {
        "categories": {
            "BodyPart": [
                "XR_ELBOW",
                "XR_FINGER",
                "XR_FOREARM",
                "XR_HAND",
                "XR_HUMERUS",
                "XR_SHOULDER",
                "XR_WRIST"
            ]
        },
        "one_hot": true
    }
}
```

| Feature | Categories |
|----------|------------|
| `BodyPart` | XR_ELBOW, XR_FINGER, XR_FOREARM, XR_HAND, XR_HUMERUS, XR_SHOULDER, XR_WRIST |

| Parameter | Description |
|------------|-------------|
| `categories` | Defines the valid categorical values. |
| `one_hot` | Whether categorical variables are one-hot encoded. |


> [!NOTE]
> One-hot encoding transforms categorical values into binary feature vectors, 
> where only the feature corresponding to the observed category is set to 1 and all others are set to 0.
> List of all categories are provided in the `categories` value as a dictionnary whose main entry
> `BodyPart` matches the [categories CSV file](https://github.com/heds-trm/mindful_subream/blob/main/examples/mura/MURA_README.md#multimodality)
> provided in the [dataset configuration](https://github.com/heds-trm/mindful_subream/blob/main/examples/mura/MURA_EXPERIMENT_CONFIGURATION.md#datasets).

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

# Shared Augmentation Stage

The `shared_augment` stage applies augmentations that affect shared modalities.

## Random Categorical Dropout

Randomly removes categorical information during training.

```json
{
    "type": "rand_dropout",
    "parameters": {
        "prob": 0.05,
        "feature_prob": 1.0
    }
}
```

| Parameter | Description |
|------------|-------------|
| `prob` | Probability of applying the augmentation. |
| `feature_prob` | Probability of dropping individual features once the augmentation is triggered. |

> [!NOTE]
> - This augmentation with a 5% chance of occuring encourages the model to remain robust when metadata is missing or incomplete.
> - Since `feature_prob = 1.0`, all categorical features are removed whenever the augmentation is applied.

---

# View Augmentation Stage

The `view_augment` stage applies image augmentations during training and they are identical to the 
[unimodal mode](https://github.com/heds-trm/mindful_core/blob/main/examples/mura/MURA_PIPELINE_CONFIGURATION.md#data-augmentation-stage):
- Random flip
- Random rotation
- Random translation
- Salt-and-Pepper and Gaussian noises addition

---

# Pipeline Summary

This pipeline performs the following operations:

1. Load and preprocess image data.
2. Convert images to grayscale.
3. Resize all images to **512 × 512**.
4. Normalize image intensities.
5. Convert images and labels to tensors.
6. One-hot encode categorical metadata.
7. Randomly remove categorical features during training to improve robustness.
8. Apply image augmentations:
   - Random flipping
   - Random rotations
   - Random translations
   - Salt-and-pepper noise
   - Gaussian noise

The resulting pipeline enables multimodal learning by combining imaging information with structured metadata while improving model generalization through data augmentation.
