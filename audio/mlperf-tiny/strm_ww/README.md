# Streaming Wake Word

MLPerf Tiny streaming wake word model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | wake word detection |
| Model artifact | `audio/mlperf-tiny/strm_ww/model.tflite` |
| Golden fixture | `audio/mlperf-tiny/strm_ww/golden.npz` |
| Inputs | 1 tensor, `[1, 30, 1, 40]`, `int8` |
| Outputs | 1 tensor, `[1, 3]`, `int8` |
| Precision | int8 |
| Source | `TODO(verify)`: the bytes do not match mlcommons/tiny `streaming_wakeword/trained_models/str_ww_ref_model.tflite` (its only revision, `904de6f8`) |
| License | `TODO(verify)`; the MLPerf Tiny reference model is Apache-2.0 |
| Training data | `TODO(verify)`; the MLPerf Tiny reference trains on Google Speech Commands v2 with MUSAN noise |
| Conversion | `TODO(verify)` |
| Notes | Includes a checked-in golden fixture in the model directory |
