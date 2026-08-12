# RNNoise

RNNoise speech denoising model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech denoising |
| Model artifact | `audio/rnnoise/model.tflite` |
| Golden fixture | `audio/rnnoise/golden.npz` |
| Inputs | 4 tensors: `[1, 1, 42]`, `[1, 24]`, `[1, 48]`, `[1, 96]`, all `int8` |
| Outputs | 5 tensors: `[1, 1, 96]`, `[1, 1, 22]`, `[1, 1, 48]`, `[1, 1, 24]`, `[1, 1, 1]`, all `int8` |
| Precision | int8 |
| Provenance | Prebuilt RNNoise-derived TFLite artifact checked into this repo |
| Notes | Includes recurrent state inputs and recurrent state outputs in the golden fixture |
