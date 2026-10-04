# Wav2Letter

Pruned Wav2Letter speech recognition model stored as a prebuilt int8 TFLite artifact.

| Field | Value |
| --- | --- |
| Domain | audio |
| Benchmark family | standalone |
| Task | speech recognition |
| Inputs | 1 tensor, `[1, 296, 39]`, `int8` |
| Outputs | 1 tensor, `[1, 1, 148, 29]`, `int8` |
| Source | [ARM-software/ML-zoo `speech_recognition/wav2letter/tflite_pruned_int8/wav2letter_pruned_int8.tflite`](https://github.com/ARM-software/ML-zoo/blob/1a92aa08c0de49a7304e0a7f3f59df6f4fd33ac8/models/speech_recognition/wav2letter/tflite_pruned_int8/wav2letter_pruned_int8.tflite) at `1a92aa08`, byte-identical (later upstream revisions differ) |
| License | Apache-2.0 ([text](../../licenses/Apache-2.0.txt)), per the ML-zoo model README |
| Training | Created by Arm, pruned to 50% sparsity and fine-tuned, per the ML-zoo model README, which lists LibriSpeech as its dataset; dataset terms `TODO(verify)` |
| Conversion | Upstream int8 quantization with the TensorFlow Model Optimization Toolkit; the artifact is unchanged |
| Notes | Includes a checked-in golden fixture in the model directory |
