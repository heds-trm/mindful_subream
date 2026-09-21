# Pipeline Configuration
 
The pipelines of the ViT-Transformer example are **identical to the [ViT-LSTM pipelines](../ViT-LSTM/ViT-LSTM_PIPELINE_CONFIGURATION.md)**: the 4D image is split into one modality per temporal phase (`phase_0`, `phase_2`, ...) with `take_slice` operations, since the ViT-Transformer encoder also expects each phase as a separate input.
 
Please refer to the [ViT-LSTM pipeline configuration](../ViT-LSTM/ViT-LSTM_PIPELINE_CONFIGURATION.md) for the full description of the preprocessing and augmentation stages.
 
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
> The output modality names must exactly match the `encoders_configs` entries of the [encoder hparams](./ViT-Transformer_HPARAMS_CONFIGURATION.md#vit-transformer-encoder-configuration) used by the experiment.
 
# Pipeline Summary
 
The pipeline performs the following operations (see the [ViT-LSTM pipeline configuration](../ViT-LSTM/ViT-LSTM_PIPELINE_CONFIGURATION.md) for details)
