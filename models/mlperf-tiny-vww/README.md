# Visual Wake Word

MLPerf Tiny visual wake word reference model stored as TFLite artifacts in four precisions: `fp32`, `fp16`, `a8w8` and `a16w8`.

| Field | Value |
| --- | --- |
| Domain | vision |
| Benchmark family | MLPerf Tiny |
| Task | visual wake word detection |
| Inputs | 1 tensor, `[1, 96, 96, 3]`, `int8` (`a8w8`), `int16` (`a16w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 2]`, `int8` (`a8w8`), `int16` (`a16w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `visual_wake_words/trained_models/vww_96_int8.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite) at `4addd0fa`, byte-identical |
| fp32 source | [mlcommons/tiny `visual_wake_words/trained_models/vww_96_float.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96_float.tflite) at `4addd0fa`, byte-identical; its golden is generated here (seed 42, LiteRT reference kernels) |
| fp16 | Built with helia-edge 0.6.2 `export()` (commit `260473bf`, TensorFlow 2.21.0) from [`visual_wake_words/trained_models/vww_96.h5`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96.h5) at `4addd0fa`: a native float16 graph (float16 inputs, weights, activations and outputs). LiteRT has no float16 kernels, so there is no fp16 golden; compare against the fp32 golden, whose inputs cast to float16 are this model's inputs |
| a16w8 | Built with helia-edge `export()` with `ExportOptions(dense_per_channel=False)` ([AmbiqAI/helia-edge#105](https://github.com/AmbiqAI/helia-edge/pull/105), merged as `64a7410c`; TensorFlow 2.21.0) from [`visual_wake_words/trained_models/vww_96.h5`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96.h5) at `4addd0fa`: int16 activations and int8 weights. Convolution weights are per-channel and FULLY_CONNECTED weights per-tensor, so upstream LiteRT for Microcontrollers (CMSIS-NN) runs it as well as LiteRT. Calibration: COCO val2017, the first 200 images of MLPerf Tiny's vww01 list (all labelled person), 96x96 bilinear, /255; sha256 `ebfc082a5e42704da23cdce7c9a9b1d5957a857c82a63c6e7f9932d0cef4fd84`. Golden: seed 42, LiteRT reference kernels |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/visual_wake_words`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words)); data: Visual Wake Words derived from COCO 2014 (`vw_coco2014_96`); dataset terms `TODO(verify)` |
| Conversion | None for the upstream artifacts (`a8w8` and `fp32`) as published; the other precisions are built here (rows above) |
| Notes | Includes a checked-in golden fixture for each precision except `fp16` |
