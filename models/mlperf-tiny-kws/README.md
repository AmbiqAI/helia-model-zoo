# KWS Reference

MLPerf Tiny keyword spotting reference model stored as TFLite artifacts in three precisions: `fp32`, `fp16` and `a8w8`.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | keyword spotting |
| Inputs | 1 tensor, `[1, 49, 10, 1]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Outputs | 1 tensor, `[1, 12]`, `int8` (`a8w8`), `float16` (`fp16`) or `float32` (`fp32`) |
| Source | [mlcommons/tiny `keyword_spotting/trained_models/kws_ref_model.tflite`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite) at `4addd0fa`, byte-identical |
| fp32 | Built with helia-edge 0.6.2 `export()` (commit `260473bf`, TensorFlow 2.21.0) from MLPerf Tiny's DS-CNN (`keyword_spotting/keras_model.py`, `ds_cnn`) with the float weights of [`keyword_spotting/trained_models/kws_ref_model`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model) at `4addd0fa`. Upstream's `kws_ref_model_float32.tflite` stores int8 convolution weights, so it is not used. 91.3% on 600 Speech Commands v0.02 test clips. Golden: seed 42, LiteRT reference kernels |
| fp16 | Built with helia-edge 0.6.2 `export()` (commit `260473bf`, TensorFlow 2.21.0) from MLPerf Tiny's DS-CNN (`keyword_spotting/keras_model.py`, `ds_cnn`) with the float weights of [`keyword_spotting/trained_models/kws_ref_model`](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting/trained_models/kws_ref_model) at `4addd0fa`: a native float16 graph (float16 inputs, weights, activations and outputs). LiteRT has no float16 kernels, so there is no fp16 golden; compare against the fp32 golden, whose inputs cast to float16 are this model's inputs |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/keyword_spotting`](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/keyword_spotting)); data: Google Speech Commands v2; dataset terms `TODO(verify)` |
| Conversion | None for the upstream artifacts (`a8w8`) as published; the other precisions are built here (rows above) |
| Notes | Includes a checked-in golden fixture for each precision except `fp16` |
