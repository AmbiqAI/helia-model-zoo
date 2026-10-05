# ResNet

MLPerf Tiny image classification reference model stored as the prebuilt int8 (`a8w8`) and float32 (`fp32`) TFLite artifacts.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | image classification |
| Inputs | 1 tensor, `[1, 32, 32, 3]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 10]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `image_classification/trained_models/pretrainedResnet_quant.tflite`](https://github.com/mlcommons/tiny/blob/eb78d0ebaf2c812ce13668f017a22171a38cd051/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite) at `eb78d0eb`, byte-identical |
| fp32 source | [mlcommons/tiny `image_classification/trained_models/pretrainedResnet.tflite`](https://github.com/mlcommons/tiny/blob/eb78d0ebaf2c812ce13668f017a22171a38cd051/benchmark/training/image_classification/trained_models/pretrainedResnet.tflite) at `eb78d0eb`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/image_classification`](https://github.com/mlcommons/tiny/tree/eb78d0ebaf2c812ce13668f017a22171a38cd051/benchmark/training/image_classification)); data: CIFAR-10; dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published (later upstream revisions differ) |
| Notes | Includes a checked-in golden fixture for each precision |
