# KWS Reference

MLPerf Tiny keyword spotting reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | keyword spotting |
| Model artifact | `audio/mlperf-tiny/kws_ref/model.tflite` |
| Golden fixture | `audio/mlperf-tiny/kws_ref/golden.npz` |
| Inputs | 1 tensor, `[1, 49, 10, 1]`, `int8` |
| Outputs | 1 tensor, `[1, 12]`, `int8` |
| Precision | int8 |
| Source | [mlcommons/tiny `keyword_spotting/trained_models/kws_ref_model.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite) at `4addd0fa`, byte-identical |
| License | Apache-2.0 ([text](../../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training data | Google Speech Commands v2, per the upstream training README; dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published |
| Notes | Includes a checked-in golden fixture in the model directory |
