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

Two manifests pin every model/golden pair by SHA-256:

- `src/helia_model_zoo/manifest.json` (manifest v2) has one entry per model ID
  with its precisions, tensors, state pairs, golden metadata, license and
  upstream source. It ships with the Python package.
- `corpus-manifest-v1.json` keeps the v1 format that helia-aot reads. Each v1 ID
  is an alias of a v2 entry, and CI checks that the two agree.

CI hydrates Git LFS and checks artifact hashes, the declared tensors, NPZ keys,
shapes and dtypes against each TFLite model. It also checks that each golden's
outputs replay exactly under its recorded LiteRT resolver and version
(`tools/golden-requirements.txt` pins that version). Run the same checks locally
from a hydrated checkout:

```bash
python -m pip install -r tools/golden-requirements.txt
python -m pip install --no-deps -e .
python tools/validate_corpus.py corpus-manifest-v1.json
helia-zoo validate --replay
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

## License

The code, documentation and tooling here are licensed under the
[BSD 3-Clause License](LICENSE). Model artifacts keep the license of their
upstream source, and each golden fixture follows its model; [NOTICE](NOTICE)
lists them, and each model card records its source, license, training and
conversion.

## Python package

```bash
python -m pip install "helia-model-zoo[litert] @ git+https://github.com/AmbiqAI/helia-model-zoo@<commit>"
```

```python
import helia_model_zoo as zoo

entry = zoo.get("rnnoise")         # an ID or a v1 alias such as "rnnoise-int8"
entry.precisions["int8"].inputs    # names, shapes, dtypes, scales, zero points
entry.io.state_pairs               # explicit state: which output feeds which input

model = entry.fetch("int8")        # a local path whose sha256 matches the manifest
golden = entry.golden("int8")      # golden.inputs / golden.outputs as NumPy arrays
zoo.resolve("zoo://rnnoise/int8")  # the same model by URI
```

`helia-zoo list`, `show <id>`, `fetch <id> [--golden] [--card]` and `validate`
expose the same on the command line. Installing from Git with git-lfs present
downloads every artifact in the repository; set `GIT_LFS_SKIP_SMUDGE=1` to skip
them.

Files are fetched as follows:

- Files in this repository are read from a hydrated checkout when there is one
  (`root=`, `HELIA_ZOO_ROOT`, or an editable install). Otherwise they are
  downloaded from GitHub at the commit the package was installed from
  (`HELIA_ZOO_REVISION` overrides it).
- `https://` sources are downloaded directly.
- `hf://` sources (`hf://[datasets/]<org>/<repo>@<40-hex commit>/<path>`) need
  the `hf` extra and use huggingface_hub's own login (`HF_TOKEN` or
  `hf auth login`).
- Every model, golden or other file with a sha256 is checked for size and
  sha256 before it enters the cache (`HELIA_ZOO_CACHE`, default
  `~/.cache/helia-model-zoo`), and checked again on every fetch. A card or
  overlay manifest without a sha256 is pinned by its revision instead.

### Private entries

Private models never enter this repository. Their entries live in an overlay
manifest that only its users can read; `HELIA_ZOO_OVERLAY` names it (a local
path or an `hf://` URI, read once per process), and its entries join the
packaged ones. Before pushing
any change here, run

```bash
helia-zoo guard --overlay "$HELIA_ZOO_OVERLAY" --text pr-body.md
```

It refuses if any overlay ID, alias, title, Hugging Face repository, upstream
or sha256 appears in what a push would publish: the content, paths, messages
and authors of every commit since `origin/main`, the index and working tree
(including file names and symlink targets), the branch name, or the extra text
files. Names that the manifest at `origin/main` also uses are public and are not
reported.

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
