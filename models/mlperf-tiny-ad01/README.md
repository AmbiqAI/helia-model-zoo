# AD01

MLPerf Tiny anomaly detection reference model stored as TFLite artifacts in three precisions: `fp32`, `fp16` and `a8w8`.

| Field | Value |
| --- | --- |
| Domain | anomaly-detection |
| Benchmark family | MLPerf Tiny |
| Task | anomaly detection |
| Inputs | 1 tensor, `[1, 640]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 640]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `anomaly_detection/trained_models/ad01_int8.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite) at `4addd0fa`, byte-identical |
| fp32 source | [mlcommons/tiny `anomaly_detection/trained_models/ad01_fp32.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_fp32.tflite) at `4addd0fa`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| fp16 | Built with helia-edge 0.6.2 `export()` (commit `260473bf`, TensorFlow 2.21.0) from [`anomaly_detection/trained_models/ad01.h5`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01.h5) at `4addd0fa`: a native float16 graph (float16 inputs, weights, activations and outputs). LiteRT has no float16 kernels, so there is no fp16 golden; its fp32 counterpart is upstream's `ad01.h5`, whose outputs differ from upstream's `ad01_fp32.tflite` (this record's fp32) by up to 4.2 (MSE 0.15) on outputs of RMS 31 on ToyCar vectors, so the fp32 golden is a reference only to that tolerance |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/anomaly_detection`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection)); data: DCASE 2020 Challenge Task 2 (anomalous machine sounds); dataset terms `TODO(verify)` |
| Conversion | None for the upstream artifacts (`a8w8` and `fp32`) as published; the other precisions are built here (rows above) |
| Notes | Includes a checked-in golden fixture for each precision except `fp16` |
