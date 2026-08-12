# Wav2Letter

Pruned Wav2Letter speech recognition model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech recognition |
| Model artifact | `audio/wav2letter/model.tflite` |
| Golden fixture | `audio/wav2letter/golden.npz` |
| Inputs | 1 tensor, `[1, 296, 39]`, `int8` |
| Outputs | 1 tensor, `[1, 1, 148, 29]`, `int8` |
| Precision | int8 |
| Provenance | Prebuilt pruned Wav2Letter TFLite artifact checked into this repo |
| License review | Upstream provenance must be reviewed before redistribution; inclusion here is not a new license grant |
| Notes | Includes a checked-in golden fixture in the model directory |
