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

To contribute a model/golden pair for helia-aot release testing, follow
[Add a model to the release corpus](docs/how-to/add-release-model.md). Adding an
artifact here and enabling it in helia-aot are separate, reviewed changes.

The template in `convert-yaml/convert.yaml` is a `heliaAOT` conversion template. It is included as a reference for adapting a zoo model into a `heliaAOT` conversion flow with a custom module output path or platform configuration.

## Golden fixtures

Golden fixtures are stored as `.npz` files with a stable key layout:

- `input_0`, `input_1`, ...
- `output_0`, `output_1`, ...

`corpus-manifest-v1.json` is the machine-readable source of artifact identity.
It pins every established model/golden pair by SHA-256 and records the runtime
and model-card references used for provenance and licensing review. CI hydrates
Git LFS and validates artifact hashes, NPZ keys, shapes, and dtypes against the
TFLite signature. Run the same preflight locally with:

```bash
python tools/validate_corpus.py corpus-manifest-v1.json
```

The per-model README referenced by each manifest entry is the artifact's model
card. For third-party models, it must identify the upstream source and the
applicable upstream license; inclusion in this repository is not a new license
grant. Do not infer a license solely from a model-family name.

### Reproducing future golden changes

Future golden updates use the pinned LiteRT environment in
`tools/golden-requirements.txt`, never helia-aot itself:

```bash
python -m venv .golden-venv
. .golden-venv/bin/activate
python -m pip install -r tools/golden-requirements.txt
python tools/generate_golden.py path/to/model.tflite path/to/golden.npz --seed 42
python tools/validate_corpus.py corpus-manifest-v1.json
```

After intentionally changing a golden, update its manifest digest. The review
description must state the reference runtime/version, seed and any non-default
input generation, changed outputs, representative and maximum numerical
differences from the prior fixture, and approval from the model/corpus owner.
Release goldens must not be generated with helia-aot.

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
