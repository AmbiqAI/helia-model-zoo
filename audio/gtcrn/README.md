# GTCRN

GTCRN streaming speech enhancement model stored as a prebuilt int16 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech enhancement |
| Model artifact | `audio/gtcrn/model.tflite` |
| Golden fixture | Not generated |
| Inputs | 1 tensor, `[1, 129, 1, 3]`, `int16` |
| Outputs | 1 tensor, `[1, 129, 1, 2]`, `int16` |
| Precision | int16 |
| Source | Derived from GTCRN ([Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)); the exported checkpoint and conversion are `TODO(verify)`; the upstream repository has no TFLite file to compare |
| License | `TODO(verify)` for the derived artifact; upstream is MIT ([text](../../licenses/gtcrn-MIT.txt)) |
| Training | `TODO(verify)` |
| Conversion | int16 conversion; recipe `TODO(verify)` |
| Notes | No checked-in golden fixture is currently included for this model |
