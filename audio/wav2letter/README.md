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
| Source | `TODO(verify)`: the bytes match no revision of ML-zoo `wav2letter`, `wav2letter_pruned` or `tiny_wav2letter` int8 |
| License | `TODO(verify)` |
| Training data | `TODO(verify)` |
| Conversion | `TODO(verify)` |
| Notes | Includes a checked-in golden fixture in the model directory |
