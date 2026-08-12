# Visual Wake Word

MLPerf Tiny visual wake word reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | visual wake word detection |
| Model artifact | `vision/mlperf-tiny/vww/model.tflite` |
| Golden fixture | `vision/mlperf-tiny/vww/golden.npz` |
| Inputs | 1 tensor, `[1, 96, 96, 3]`, `int8` |
| Outputs | 1 tensor, `[1, 2]`, `int8` |
| Precision | int8 |
| Provenance | MLPerf Tiny reference model shipped in this repo as a prebuilt TFLite file |
| Notes | Includes a checked-in golden fixture in the model directory |
