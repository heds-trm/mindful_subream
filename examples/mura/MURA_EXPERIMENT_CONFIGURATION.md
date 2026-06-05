
# Experiment Configuration

Experiments are defined through JSON configuration files. A configuration specifies:

- Available datasets
- Shared training settings
- Experiment definitions
- Logging locations

This approach allows experiments to be reproduced and modified without changing the source code.

> [!CAUTION]
> According to JSON specification, you cannot add comments and trailing commas in a .json file leading to possible crashes.
>
> Ex. {"one":1,"two":2,} is forbidden.

Most of the configuration in the [multimodal classification example](./MURA_README.md) with the MURA dataset is similar to the [unimodal configuration](https://github.com/heds-trm/mindful_core/blob/main/examples/mura/MURA_EXPERIMENT_CONFIGURATION.md). We highlight here the key differences.

## Example Configuration

```json
{
    "datasets": {
        "mura_dataset": {
            "folds": "<folds_dir>",
            "categorical_features": "<folds_dir>/mura_body_parts.csv"
        }
    },
    {
        "multimodal_densenet121_mura": {
            "dataset": "mura_dataset",
            "folds": "all",
            "model": {
                "class_name": "sieve",
                "hparams": "<models_dir>/multimodal_densenet121_hparams.json"
            },
            "pipeline_config": "<pipelines_dir>/multimodal_mura.json",
            "stages": "train test"
        }
    }
    "log_dir": "<logs_dir>"
}
````

---

# Datasets

The `datasets` section defines the datasets available to experiments. The `categorical_features` parameter specifies a file containing non-imaging categorical features associated with each sample. In our case, which [body part the radiograph belongs to](./MURA_README.md#multimodality).

# Experiments

The `experiments` section defines the experiments to execute and is pretty similar to the unimodal counterpart. The main difference is that our "Sieve" model is an architecture able to fuse multimodal data, requiring specific [hyperparameters](./MURA_HPARAMS_CONFIGURATION.md). 

Similarly, the pipeline to process multimodal data needs a dedicated [configuration file](./MURA_PIPELINE_CONFIGURATION.md).

