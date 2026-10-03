# Streaming Wake Word

MLPerf Tiny streaming wake word model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | MLPerf Tiny |
| Task | wake word detection |
| Inputs | 1 tensor, `[1, 30, 1, 40]`, `int8` |
| Outputs | 1 tensor, `[1, 3]`, `int8` |
| Source | [mlcommons/tiny `streaming_wakeword/trained_models/strm_ww_int8.tflite`](https://github.com/mlcommons/tiny/blob/8e4803009a3be9aede6de47062f185c9eb97d126/benchmark/training/streaming_wakeword/trained_models/strm_ww_int8.tflite) at `8e480300`, byte-identical; upstream later replaced it with `str_ww_ref_model.tflite` |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), from MLPerf Tiny |
| Training | MLPerf Tiny reference recipe ([`benchmark/training/streaming_wakeword`](https://github.com/mlcommons/tiny/tree/8e4803009a3be9aede6de47062f185c9eb97d126/benchmark/training/streaming_wakeword)); data: Google Speech Commands v2 with MUSAN noise; dataset terms `TODO(verify)` |
| Conversion | None; the upstream int8 artifact as published |
| Notes | Includes a checked-in golden fixture in the model directory |
