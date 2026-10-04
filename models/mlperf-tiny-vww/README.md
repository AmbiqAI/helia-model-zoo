# Visual Wake Word

MLPerf Tiny visual wake word reference model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | visual wake word detection |
| Inputs | 1 tensor, `[1, 96, 96, 3]`, `int8` |
| Outputs | 1 tensor, `[1, 2]`, `int8` |
| Source | [mlcommons/tiny `visual_wake_words/trained_models/vww_96_int8.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite) at `4addd0fa`, byte-identical |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/visual_wake_words`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words)); data: Visual Wake Words derived from COCO 2014 (`vw_coco2014_96`); dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published |
| Notes | Includes a checked-in golden fixture in the model directory |
