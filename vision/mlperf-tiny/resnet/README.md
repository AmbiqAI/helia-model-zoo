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
| Source | [mlcommons/tiny `image_classification/trained_models/pretrainedResnet_quant.tflite`](https://github.com/mlcommons/tiny/blob/eb78d0ebaf2c812ce13668f017a22171a38cd051/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite) at `eb78d0eb`, byte-identical |
| License | Apache-2.0 ([text](../../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training data | CIFAR-10, per the upstream training README; dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published (later upstream revisions differ) |
| Notes | Includes a checked-in golden fixture in the model directory |
