# ResNet

MLPerf Tiny image classification reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | image classification |
| Model artifact | `vision/mlperf-tiny/resnet/model.tflite` |
| Golden fixture | `vision/mlperf-tiny/resnet/golden.npz` |
| Inputs | 1 tensor, `[1, 32, 32, 3]`, `int8` |
| Outputs | 1 tensor, `[1, 10]`, `int8` |
| Precision | int8 |
| Provenance | MLPerf Tiny reference model shipped in this repo as a prebuilt TFLite file |
| Notes | Includes a checked-in golden fixture in the model directory |
