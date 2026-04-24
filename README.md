# heliaAOT Model Zoo

This repository is a model zoo for prebuilt TFLite models. The repo groups models by domain, keeps golden input/output fixtures next to those domains when available, and provides lightweight documentation for each model entry.

## Repository layout

```text
<domain>/<family?>/<model>/
  README.md
  golden.npz
  model.tflite
```

Each model directory is self-contained: the TFLite artifact, the checked-in golden fixture when available, and the model card all live together.

The template in `convert-yaml/convert.yaml` is a `heliaAOT` conversion template. It is included as a reference for adapting a zoo model into a `heliaAOT` conversion flow with a custom module output path or platform configuration.

## Golden fixtures

Golden fixtures are stored as `.npz` files with a stable key layout:

- `input_0`, `input_1`, ...
- `output_0`, `output_1`, ...

## Domains

- [Audio](audio/README.md)
- [Vision](vision/README.md)
- [Anomaly Detection](anomaly-detection/README.md)

## Inventory

| Model | Domain | Family | Task | Quantization | Model | Golden | Docs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AD01 | anomaly-detection | MLPerf Tiny | anomaly detection | int8 | [model](anomaly-detection/mlperf-tiny/ad01/model.tflite) | [golden](anomaly-detection/mlperf-tiny/ad01/golden.npz) | [docs](anomaly-detection/mlperf-tiny/ad01/README.md) |
| KWS Reference | audio | MLPerf Tiny | keyword spotting | int8 | [model](audio/mlperf-tiny/kws_ref/model.tflite) | [golden](audio/mlperf-tiny/kws_ref/golden.npz) | [docs](audio/mlperf-tiny/kws_ref/README.md) |
| Streaming Wake Word | audio | MLPerf Tiny | wake word detection | int8 | [model](audio/mlperf-tiny/strm_ww/model.tflite) | [golden](audio/mlperf-tiny/strm_ww/golden.npz) | [docs](audio/mlperf-tiny/strm_ww/README.md) |
| RNNoise | audio | standalone | speech denoising | int8 | [model](audio/rnnoise/model.tflite) | [golden](audio/rnnoise/golden.npz) | [docs](audio/rnnoise/README.md) |
| Wav2Letter | audio | standalone | speech recognition | int8 | [model](audio/wav2letter/model.tflite) | [golden](audio/wav2letter/golden.npz) | [docs](audio/wav2letter/README.md) |
| DFNet2 | audio | standalone | speech enhancement | int16 | [model](audio/dfnet2/model.tflite) | [golden](audio/dfnet2/golden.npz) | [docs](audio/dfnet2/README.md) |
| GTCRN | audio | standalone | speech enhancement | int16 | [model](audio/gtcrn/model.tflite) | `not included` | [docs](audio/gtcrn/README.md) |
| ResNet | vision | MLPerf Tiny | image classification | int8 | [model](vision/mlperf-tiny/resnet/model.tflite) | [golden](vision/mlperf-tiny/resnet/golden.npz) | [docs](vision/mlperf-tiny/resnet/README.md) |
| Visual Wake Word | vision | MLPerf Tiny | visual wake word detection | int8 | [model](vision/mlperf-tiny/vww/model.tflite) | [golden](vision/mlperf-tiny/vww/golden.npz) | [docs](vision/mlperf-tiny/vww/README.md) |
| MobileNet V2 1.0 224 | vision | standalone | image classification | int8 | [model](vision/mobilenet_v2/model.tflite) | [golden](vision/mobilenet_v2/golden.npz) | [docs](vision/mobilenet_v2/README.md) |
