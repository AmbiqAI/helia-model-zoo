# KWS Reference

MLPerf Tiny keyword spotting reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | keyword spotting |
| Model artifact | `audio/mlperf-tiny/kws_ref/model.tflite` |
| Config | `audio/mlperf-tiny/kws_ref/config.yaml` |
| Golden fixture | `audio/mlperf-tiny/kws_ref/golden.npz` |
| Inputs | 1 tensor, `[1, 49, 10, 1]`, `int8` |
| Outputs | 1 tensor, `[1, 12]`, `int8` |
| Precision | int8 |
| Provenance | MLPerf Tiny reference model shipped in this repo as a prebuilt TFLite file |
| Notes | Includes a checked-in golden fixture in the model directory |
