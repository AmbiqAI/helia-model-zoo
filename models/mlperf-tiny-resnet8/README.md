# ResNet-8

The MLPerf Tiny image classification reference model, ResNet-8 (78,666 parameters, convolution widths 16/32/64), stored as TFLite artifacts in three precisions: `fp32`, `fp16` and `a8w8`.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | image classification |
| Inputs | 1 tensor, `[1, 32, 32, 3]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 10]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `image_classification/trained_models/pretrainedResnet_quant.tflite`](https://github.com/mlcommons/tiny/blob/1bbcf37fac1b0c4be0bada2d7ddfb1e0bb4cb6a0/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite) at `1bbcf37f`, byte-identical |
| fp32 source | [mlcommons/tiny `image_classification/trained_models/pretrainedResnet.tflite`](https://github.com/mlcommons/tiny/blob/1bbcf37fac1b0c4be0bada2d7ddfb1e0bb4cb6a0/benchmark/training/image_classification/trained_models/pretrainedResnet.tflite) at `1bbcf37f`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| fp16 | Built with helia-edge 0.6.2 `export()` (commit `260473bf`, TensorFlow 2.21.0) from [`image_classification/trained_models/pretrainedResnet.h5`](https://github.com/mlcommons/tiny/blob/1bbcf37fac1b0c4be0bada2d7ddfb1e0bb4cb6a0/benchmark/training/image_classification/trained_models/pretrainedResnet.h5) at `1bbcf37f`: a native float16 graph (float16 inputs, weights, activations and outputs). LiteRT has no float16 kernels, so there is no fp16 golden; compare against the fp32 golden, whose inputs cast to float16 are this model's inputs |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/image_classification`](https://github.com/mlcommons/tiny/tree/1bbcf37fac1b0c4be0bada2d7ddfb1e0bb4cb6a0/benchmark/training/image_classification)); data: CIFAR-10; dataset terms `TODO(verify)` |
| Conversion | None for the upstream artifacts (`a8w8` and `fp32`) as published; the other precisions are built here (rows above) |
| Notes | Includes a checked-in golden fixture for each precision except `fp16` |
