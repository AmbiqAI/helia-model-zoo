# AD01

MLPerf Tiny anomaly detection reference model stored as the prebuilt int8 (`a8w8`) and float32 (`fp32`) TFLite artifacts.

| Field | Value |
| --- | --- |
| Domain | anomaly-detection |
| Benchmark family | MLPerf Tiny |
| Task | anomaly detection |
| Inputs | 1 tensor, `[1, 640]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 640]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `anomaly_detection/trained_models/ad01_int8.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite) at `4addd0fa`, byte-identical |
| fp32 source | [mlcommons/tiny `anomaly_detection/trained_models/ad01_fp32.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection/trained_models/ad01_fp32.tflite) at `4addd0fa`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/anomaly_detection`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/anomaly_detection)); data: DCASE 2020 Challenge Task 2 (anomalous machine sounds); dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 and float32 artifacts as published |
| Notes | Includes a checked-in golden fixture for each precision |
