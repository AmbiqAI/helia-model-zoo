# AD01

MLPerf Tiny anomaly detection reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | anomaly-detection |
| Benchmark family | MLPerf Tiny |
| Task | anomaly detection |
| Model artifact | `anomaly-detection/mlperf-tiny/ad01/model.tflite` |
| Golden fixture | `anomaly-detection/mlperf-tiny/ad01/golden.npz` |
| Inputs | 1 tensor, `[1, 640]`, `int8` |
| Outputs | 1 tensor, `[1, 640]`, `int8` |
| Precision | int8 |
| Provenance | MLPerf Tiny reference model shipped in this repo as a prebuilt TFLite file |
| Notes | Includes a checked-in golden fixture in the model directory |
