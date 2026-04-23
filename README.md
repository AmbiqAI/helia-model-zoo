# Helia AOT Model Zoo

This repository is a model zoo for prebuilt TFLite models that are exercised through Helia AOT example configs. The repo groups models by domain, keeps golden input/output fixtures next to those domains, and provides lightweight documentation for each model entry.

## Repository layout

```text
<domain>/<family?>/<model>/
  README.md
  config.yaml
  golden.npz
  model.tflite
```

Each model directory is self-contained: the TFLite artifact, the Helia AOT example config, the checked-in golden fixture when available, and the model card all live together.

## Using a model config with Helia AOT

Each `config.yaml` under the domain folders is a ready-to-use example config that points at a checked-in model artifact and a checked-in golden fixture. Use the config path that matches the model you want to evaluate in your normal Helia AOT conversion or test flow.

The conversion template in `convert-yaml/convert.yaml` remains available as a reference when you need to adapt a zoo model into a custom module output path or platform configuration.

## Golden fixtures

Golden fixtures are stored as `.npz` files with a stable key layout:

- `input_0`, `input_1`, ...
- `output_0`, `output_1`, ...

## Domains

- [Audio](audio/README.md)
- [Vision](vision/README.md)
- [Anomaly Detection](anomaly-detection/README.md)

## Inventory

| Model | Domain | Family | Task | Quantization | Model | Config | Golden | Docs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AD01 | anomaly-detection | MLPerf Tiny | anomaly detection | int8 | [model](anomaly-detection/mlperf-tiny/ad01/model.tflite) | [config](anomaly-detection/mlperf-tiny/ad01/config.yaml) | [golden](anomaly-detection/mlperf-tiny/ad01/golden.npz) | [docs](anomaly-detection/mlperf-tiny/ad01/README.md) |
| KWS Reference | audio | MLPerf Tiny | keyword spotting | int8 | [model](audio/mlperf-tiny/kws_ref/model.tflite) | [config](audio/mlperf-tiny/kws_ref/config.yaml) | [golden](audio/mlperf-tiny/kws_ref/golden.npz) | [docs](audio/mlperf-tiny/kws_ref/README.md) |
| Streaming Wake Word | audio | MLPerf Tiny | wake word detection | int8 | [model](audio/mlperf-tiny/strm_ww/model.tflite) | [config](audio/mlperf-tiny/strm_ww/config.yaml) | [golden](audio/mlperf-tiny/strm_ww/golden.npz) | [docs](audio/mlperf-tiny/strm_ww/README.md) |
| RNNoise | audio | standalone | speech denoising | int8 | [model](audio/rnnoise/model.tflite) | [config](audio/rnnoise/config.yaml) | [golden](audio/rnnoise/golden.npz) | [docs](audio/rnnoise/README.md) |
| Wav2Letter | audio | standalone | speech recognition | int8 | [model](audio/wav2letter/model.tflite) | [config](audio/wav2letter/config.yaml) | [golden](audio/wav2letter/golden.npz) | [docs](audio/wav2letter/README.md) |
| DFNet2 | audio | standalone | speech enhancement | int16 | [model](audio/dfnet2/model.tflite) | [config](audio/dfnet2/config.yaml) | [golden](audio/dfnet2/golden.npz) | [docs](audio/dfnet2/README.md) |
| GTCRN | audio | standalone | speech enhancement | int16 | [model](audio/gtcrn/model.tflite) | [config](audio/gtcrn/config.yaml) | `unavailable: custom GRU op` | [docs](audio/gtcrn/README.md) |
| ResNet | vision | MLPerf Tiny | image classification | int8 | [model](vision/mlperf-tiny/resnet/model.tflite) | [config](vision/mlperf-tiny/resnet/config.yaml) | [golden](vision/mlperf-tiny/resnet/golden.npz) | [docs](vision/mlperf-tiny/resnet/README.md) |
| Visual Wake Word | vision | MLPerf Tiny | visual wake word detection | int8 | [model](vision/mlperf-tiny/vww/model.tflite) | [config](vision/mlperf-tiny/vww/config.yaml) | [golden](vision/mlperf-tiny/vww/golden.npz) | [docs](vision/mlperf-tiny/vww/README.md) |
| MobileNet V2 1.0 224 | vision | standalone | image classification | int8 | [model](vision/mobilenet_v2/model.tflite) | [config](vision/mobilenet_v2/config.yaml) | [golden](vision/mobilenet_v2/golden.npz) | [docs](vision/mobilenet_v2/README.md) |
