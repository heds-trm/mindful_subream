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


The structure of the configuration file is identical to the [Ultra example](../Ultra/Ultra_EXPERIMENT_CONFIGURATION.md). Only the referenced hparams and pipeline files differ, as the ViT-Transformer model uses its own [hyperparameters](./ViT-Transformer_HPARAMS_CONFIGURATION.md) and [pipeline](./ViT-Transformer_PIPELINE_CONFIGURATION.md).

## Example Configuration

The config experiment example configuration : [vit-transformer_config_run.json](./configs/runs/vit-transformer_config_run.json)


 
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
            "patience": 10,
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
                        "hparams": "<models_dir>/vit-transformer_hparams.json"
                    }
                }
            },
            "pipeline_config": "<pipelines_dir>/4d_pipeline.json"
        }
    },
    "log_dir": "<logs_dir>"
}
```


> [!NOTE]
> The paths shown in the run configuration files are absolute paths examples. You will need to replace them with your own paths (`<folds_dir>`, `<models_dir>`, `<pipelines_dir>`, `<logs_dir>`) according to where your data and config files are located.
 

---

# Datasets

The `datasets` section is identical to the [Ultra example](../Ultra/Ultra_EXPERIMENT_CONFIGURATION.md#datasets): one entry per classification task (`data-b_g_m`, `data-bg_m`, `data-b_m`), each pointing to the `folds` folder of the task.



# Shared Settings
 
The `shared.default` section is identical to the [Ultra example](../Ultra/Ultra_EXPERIMENT_CONFIGURATION.md#shared-settings), with one difference: the early-stopping **`patience` is set to 10** epochs instead of 25.



# Composite model
The `model` section defines the model architecture:
 
- `sub_modules.representation_model`: an `encoder` sub-module using the [ViT-Transformer encoder hparams](./ViT-Transformer_HPARAMS_CONFIGURATION.md#vit-transformer-encoder-configuration). Each temporal phase is encoded by a weight-shared 3D ViT and the sequence of phase embeddings is fused by a Transformer into a latent representation consumed by the classifier.
- `class_name`: `mlp`: multi-layer perceptron classifier, configured by an [MLP hparams file](../Ultra/Ultra_HPARAMS_CONFIGURATION.md#classifier-mlp-configuration) (2-class or 3-class variant), identical to the Ultra example.
---




# Experiments
 
The `experiments` section and its common parameters are identical to the [Ultra example](../Ultra/Ultra_EXPERIMENT_CONFIGURATION.md#experiments). Each phase-ablation experiment overrides **both** the encoder hparams and the pipeline so that the declared phase modalities match the phases extracted by the pipeline:
 
| Experiment variant | Encoder hparams | Pipeline | Phases used |
|--------------------|-----------------|----------|-------------|
| full sequence | `vit-transformer_hparams.json` | `4d_pipeline.json` | all 13 phases |
| 7 phases | `vit-transformer_hparams_7phases.json` | `4d_pipeline_7phases.json` | 0, 2, 4, 6, 8, 10, 12 |
| 4 phases | `vit-transformer_hparams_4phases.json` | `4d_pipeline_4phases.json` | 0, 4, 8, 12 |
| 3 phases | `vit-transformer_hparams_3phases.json` | `4d_pipeline_3phases.json` | 0, 6, 12 |
| 2 middle phases ("mid") | `vit-transformer_hparams_2phases_mid.json` | `4d_pipeline_2phases_mid.json` | 4, 8 |
| first and last phases ("first-last") | `vit-transformer_hparams_2phases_firstlast.json` | `4d_pipeline_2phases_firstlast.json` | 0, 12 |

 
The classifier hparams file (`mlp_2class-hparams.json` or `mlp_3class-hparams.json`) follows the classification task, as in the Ultra example.


# Logging
 
The `log_dir` parameter defines the root folder `<logs_dir>` where each experiment writes its outputs (tensorboard files, `.ckpt` checkpoints, `formatted_summary.csv`) are written to `<exp_data>/logs/<experiment_name>`.
