# KWS Reference

MLPerf Tiny keyword spotting reference model stored as the prebuilt int8 (`a8w8`) and float32 (`fp32`) TFLite artifacts.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | keyword spotting |
| Inputs | 1 tensor, `[1, 49, 10, 1]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 12]`, `int8` (`a8w8`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `keyword_spotting/trained_models/kws_ref_model.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite) at `4addd0fa`, byte-identical |
| fp32 source | [mlcommons/tiny `keyword_spotting/trained_models/kws_ref_model_float32.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model_float32.tflite) at `4addd0fa`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/keyword_spotting`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting)); data: Google Speech Commands v2; dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published |
| Notes | Includes a checked-in golden fixture for each precision |
