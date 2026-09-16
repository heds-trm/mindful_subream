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

The config experiment example configuration : [ultra_config_run.json](./4duf/configs/Ultra/configs/runs/ultra_config_run.json)


 
```json
{
    "datasets": {
        "data-b_g_m": {
            "folds": "<folds_dir>/b_g_m/"
        },
        "data-bg_m": {
            "folds": "<folds_dir>/bg_m/"
        },
        "data-b_m": {
            "folds": "<folds_dir>/b_m/"
        }
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
        "<experiment_name>": {
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
                        "hparams": "<models_dir>/ultra_hparams.json"
                    }
                }
            },
            "pipeline_config": "<pipelines_dir>/ultra-standardize.json"
        }
    },
    "log_dir": "<logs_dir>"
}
```


> [!NOTE]
> The paths shown in the run configuration files are absolute paths examples. You will need to replace them with your own paths (`<folds_dir>`, `<models_dir>`, `<pipelines_dir>`, `<logs_dir>`) according to where your data and config files are located.
 

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


# COmposite model
The `model`section defines the model architecture:

- `sub_modules.representation_model`: an `encoder` sub-module using the ["Ultra" 4D encoder hparams](./Ultra_HPARAMS_CONFIGURATION.md#ultra-encoder-configuration). It transforms the 4D image into a latent representation consumed by the classifier.
- `class_name`: `mlp`: multi-layer perceptron classifier, configured by an [MLP hparams file](./Ultra_HPARAMS_CONFIGURATION.md#classifier-mlp-configuration) (2-class or 3-class variant).

---


# Experiments
 
The `experiments` section defines the experiments to execute. Common experiment-level parameters:

| Parameter | Description |
|-----------|-------------|
| `skip` | Whether the experiment is skipped (`yes`) or run (`no`). |
| `use_imbalanced_sampler` | Enables class-imbalance-aware sampling during training (version `v2`). |
| `dataset` | Name of the dataset entry to use. |
| `folds` | Folds to run (`all` for full cross-validation). |
| `model` | Composite model (`encoder` and `mlp` with associated hparams files examples (ultra_hparams.json) and (mlp_3class-hparams.json)) |
| `pipeline_config` | Pipeline of the architecture ([pipeline config](./configs/pipelines/ultra-standardize.json)). |
| `checkpoint` | Name of another experiment whose trained checkpoint is loaded (for external external validation). |

# Logging
 
The `log_dir` parameter defines the root folder `<logs_dir>` where each experiment writes its outputs (tensorboard files, `.ckpt` checkpoints, `formatted_summary.csv`) are written to `<exp_data>/logs/<experiment_name>`.
