# Wav2Letter

Pruned Wav2Letter speech recognition model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech recognition |
| Model artifact | `audio/wav2letter/model.tflite` |
| Config | `audio/wav2letter/config.yaml` |
| Golden fixture | `audio/wav2letter/golden.npz` |
| Inputs | 1 tensor, `[1, 296, 39]`, `int8` |
| Outputs | 1 tensor, `[1, 1, 148, 29]`, `int8` |
| Precision | int8 |
| Provenance | Prebuilt pruned Wav2Letter TFLite artifact checked into this repo |
| Notes | The config keeps the existing PSRAM memory placement and FVP timeout settings |
