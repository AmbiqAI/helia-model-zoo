# MobileNet V2 1.0 224

MobileNet V2 image classification model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | standalone |
| Task | image classification |
| Model artifact | `vision/mobilenet_v2/model.tflite` |
| Golden fixture | `vision/mobilenet_v2/golden.npz` |
| Inputs | 1 tensor, `[1, 224, 224, 3]`, `int8` |
| Outputs | 1 tensor, `[1, 1001]`, `int8` |
| Precision | int8 |
| Source | [ARM-software/ML-zoo `image_classification/mobilenet_v2_1.0_224/tflite_int8/mobilenet_v2_1.0_224_INT8.tflite`](https://github.com/ARM-software/ML-zoo/blob/fec0bb5bcd486eb2839cdb7d437d466067402a6a/models/image_classification/mobilenet_v2_1.0_224/tflite_int8/mobilenet_v2_1.0_224_INT8.tflite) at `fec0bb5b`, byte-identical |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), per the ML-zoo model README |
| Training | Upstream training per the ML-zoo model README ([arXiv:1801.04381](https://arxiv.org/abs/1801.04381)); data: ILSVRC 2012 (ImageNet); dataset terms `TODO(verify)` |
| Conversion | Upstream int8 quantization; the artifact is unchanged |
| Notes | Includes a checked-in golden fixture in the model directory |
