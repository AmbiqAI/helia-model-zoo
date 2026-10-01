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
| Source | [ARM-software/ML-zoo `noise_suppression/RNNoise/tflite_int8/rnnoise_INT8.tflite`](https://github.com/ARM-software/ML-zoo/blob/fec0bb5bcd486eb2839cdb7d437d466067402a6a/models/noise_suppression/RNNoise/tflite_int8/rnnoise_INT8.tflite) at `fec0bb5b`, byte-identical |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), per the ML-zoo model README. The RNNoise algorithm ([xiph/rnnoise](https://github.com/xiph/rnnoise)) is BSD-3-Clause |
| Training data | Noisy speech database for training speech enhancement algorithms and TTS models ([Edinburgh DataShare 10283/2791](https://datashare.ed.ac.uk/handle/10283/2791)), per the ML-zoo model README |
| Conversion | Upstream post-training int8 quantization; the artifact is unchanged |
| Notes | Includes recurrent state inputs and recurrent state outputs in the golden fixture |
