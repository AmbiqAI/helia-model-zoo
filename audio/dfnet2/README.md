# DFNet2

DeepFilterNet2 streaming speech enhancement model stored as a prebuilt int16 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech enhancement |
| Model artifact | `audio/dfnet2/model.tflite` |
| Config | `audio/dfnet2/config.yaml` |
| Golden fixture | `audio/dfnet2/golden.npz` |
| Inputs | 3 tensors: `[1, 1, 1, 32]`, `[1, 1, 1, 96, 2]`, `[1, 26304]`, all `int16` |
| Outputs | 5 tensors: `[1, 1]`, `[1, 5, 1, 96, 2]`, `[1, 1]`, `[1, 1, 1, 32]`, `[1, 26304]`, all `int16` |
| Precision | int16 |
| Provenance | Prebuilt DeepFilterNet2-derived TFLite artifact checked into this repo |
| Notes | Includes recurrent state input and recurrent state outputs in the golden fixture |
