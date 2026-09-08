# Experiment Configuration

Experiments are defined through JSON configuration files. A configuration specifies:
 
- Available datasets
- Shared training settings
- Experiment definitions
- Logging locations

This approach allows experiments to be reproduced and modified without changing the source code.

> [!CAUTION]
> According to JSON specification, you cannot add comments and trailing commas in a .json file, leading to possible crashes.
>
> Ex. `{"one":1,"two":2,}` is forbidden.

## Example Configuration


 
```json
{
    "datasets": {
        "data-b_g_m": {
            "folds": "<folds_dir>/b_g_m/",
        },
        "data-b_g_m": {
            "folds": "<folds_dir>/bg_m/",
        },
        "data-b_g_m": {
            "folds": "<folds_dir>/b_m/",
        },
    },
    "shared": {
        "default": {
            "visualizers": {
                "occlusion": {
                    "image_kernel_size": 16,
                    "image_kernel_overlap": 0.6,
                    "image_kernel_sigma": 0.25,
                    "saved_versions": ["next_to_additive"],
                    "use_minimum": false
                }
            },
            "log_monitors": ["validation_loss_epoch"],
            "max_epochs": 500,
            "batch_size": 64,
            "patience": 25,
            "save_last": true,
            "num_workers": 0,
            "accelerator": "gpu",
            "devices": 1,
            "seeds": [3930787959, 3048086154, 1308332165, 4198623908, 2751205564],
            "stages": "train test"
        }
    },
    "experiments": {
        "ultra-lstm_4D_fusion_B-G-M_7": {
            "skip": "no",
            "use_imbalanced_sampler": "v2",
            "dataset": "data-b_g_m",
            "folds": "all",
            "model": {
                "class_name": "mlp",
                "hparams": "<models_dir>/mlp_3class-hparams.json",
                "sub_modules": {
                    "representation_model": {
                        "class_name": "encoder",
                        "hparams": "<models_dir>/ultra_default-hparams_best_7phases.json"
                    }
                }
            },
            "pipeline_config": "<pipelines_dir>/multimodal-ultra-standardize_7slctPhase.json"
        }
    },
    "log_dir": "<logs_dir>"
}
```


> [!NOTE]
> The paths shown in the run configuration files are absolute paths examples. You will need to replace them with your own paths (`<folds_dir>`, `<clinical_dir>`, `<models_dir>`, `<pipelines_dir>`, `<logs_dir>`) according to where your data and config files are located.
 

---

# Datasets

The `datasets` section defines the datasets available to experiments. One entry is defined **per classification task**:

| Dataset | Class grouping |
|---------|----------------|
| `data-b_g_m` | benign / lymph-node / malignant (3 classes) |
| `data-b_m` | benign / malignant (2 classes) |
| `data-bg_m` | benign+lymph-nodes / malignant (2 classes) |


Each dataset entry contains:
 
| Parameter | Description |
|------------|-------------|
| `folds` | Folder containing the cross-validation fold files for this classification task. |

# Shared Settings
 
The `shared.default` section defines settings inherited by all experiments (each experiment may override any of them):
 
| Parameter | Description |
|------------|-------------|
| `visualizers` | Post-hoc explainability. Here, occlusion maps with a kernel of size 16, 60% overlap and Gaussian smoothing (`sigma = 0.25`). |
| `log_monitors` | Metric monitored for checkpointing/early stopping (`validation_loss_epoch`). |
| `max_epochs` | Maximum number of training epochs (500). |
| `batch_size` | Training batch size (64). |
| `patience` | Early-stopping patience, in epochs (25). |
| `save_last` | Whether the last checkpoint is saved in addition to the best one. |
| `num_workers` | Number of data-loading workers. |
| `accelerator` / `devices` | Hardware configuration (1 GPU). |
| `seeds` | List of random seeds; each experiment is repeated once per seed for robustness of the results. |
| `stages` | Stages to execute (`train test`). |


# Experiments
 
revoir work in progress...


