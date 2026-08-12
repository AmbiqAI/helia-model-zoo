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
| Provenance | MLPerf Tiny reference model shipped in this repo as a prebuilt TFLite file |
| Notes | Includes a checked-in golden fixture in the model directory |
