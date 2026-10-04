# DFNet2

DeepFilterNet2 streaming speech enhancement model stored as a prebuilt int16 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech enhancement |
| Inputs | 3 tensors: `[1, 1, 1, 32]`, `[1, 1, 1, 96, 2]`, `[1, 26304]`, all `int16` |
| Outputs | 5 tensors: `[1, 1]`, `[1, 5, 1, 96, 2]`, `[1, 1]`, `[1, 1, 1, 32]`, `[1, 26304]`, all `int16` |
| Source | Derived from DeepFilterNet2 ([Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)); the exported checkpoint and conversion are `TODO(verify)`; the upstream repository has no TFLite file to compare |
| License | `TODO(verify)` for the derived artifact; upstream is MIT OR Apache-2.0 ([MIT text](../../licenses/DeepFilterNet-MIT.txt)) |
| Training | `TODO(verify)` |
| Conversion | int16 conversion; recipe `TODO(verify)` |
| Notes | Includes recurrent state input and recurrent state outputs in the golden fixture |
