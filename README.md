# heliaAOT Model Zoo

This repository is a model zoo for prebuilt TFLite models. Each model has one
small record, its card, and per precision its TFLite artifact and golden fixture.

## Repository layout

```text
models/<id>/
  record.json          helia-model-zoo/record@1
  README.md            the model card
  <precision>/model.tflite
  <precision>/golden.npz
```

The directory name is the model ID. Precisions are `fp32`, `fp16`, `a8w8`,
`a16w8` and `a8w4`. There is no other inventory: `helia-zoo list` prints the
models from the records.

To contribute a model/golden pair for helia-aot release testing, follow
[Add a model to the release corpus](docs/how-to/add-release-model.md). Adding an
artifact here and enabling it in helia-aot are separate, reviewed changes.

The template in `convert-yaml/convert.yaml` is a `heliaAOT` conversion template. It is included as a reference for adapting a zoo model into a `heliaAOT` conversion flow with a custom module output path or platform configuration.

## Golden fixtures

Golden fixtures are stored as uncompressed `.npz` files with a stable key
layout, in the model's input and output order:

- `input_0`, `input_1`, ...
- `output_0`, `output_1`, ...

Each array holds the raw values fed to or produced by the model. A golden's
`kind` is one of:

- `single`: one call;
- `batch`: `steps` independent calls, each with a leading array axis;
- `sequence`: `steps` calls of a streaming model, where each explicit state
  input takes the previous call's paired output, except at step 0 and at the
  steps listed in `resets`.

A state starts from real zero, which a quantized tensor stores as its zero
point. A quantized sequence needs each state pair to share its scale and zero
point, so the state is carried exactly.

```bash
helia-zoo golden generate model.tflite golden.npz --kind sequence --steps 64 \
    --resets 32 --data inputs.npz --print-record --path a8w8/golden.npz \
    --source-uri https://example.com/clip.wav --source-sha256 <sha256>
helia-zoo golden check model.tflite golden.npz --kind sequence --steps 64 --resets 32
```

State pairs come from a model's record (`--record <id>`), from `--pair IN:OUT`, or
from `state_in_k`/`state_out_k` signature or tensor names; a sequence without any state
pair is refused. Inputs not given with `--data` are drawn from `--seed`; when
`--print-record` records a `--data` source, `--data` must give every
non-state input. New goldens use LiteRT's reference kernels
(`builtin_ref`) unless `--resolver` says otherwise. In Python,
`helia_model_zoo.golden.check()` checks one golden file against its model
without a record.

The records pin every model and golden by SHA-256. Each one lists its
precisions, tensors, state pairs (by tensor name), golden metadata, licence and
upstream source. The records and cards ship with the Python package.
`corpus-manifest-v1.json` keeps the v1 format that helia-aot reads; CI checks
that each v1 entry agrees with the record holding its model.

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

The README in each model directory is the artifact's model
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

`tools/generate_golden.py` writes a `single` golden with LiteRT's reference
kernels; `--resolver builtin` selects the optimized kernels instead. Two
behaviours differ from earlier versions of the script:

- Outputs come from the reference kernels by default. For some models (here,
  KWS, RNNoise, Wav2Letter and MobileNet V2) they differ from the optimized
  kernels' outputs. With `--resolver builtin`, the script reproduces the
  earlier script's files.
- A model with `state_in_k`/`state_out_k` state names gets its state inputs at
  the reset value, not drawn from the seed. The other inputs' draws therefore
  differ from the earlier script for the same seed. None of the current models
  has such names.

After intentionally changing a golden, update its record's digest. The review
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

record = zoo.get("rnnoise")
record.precisions["a8w8"].inputs   # names, shapes, dtypes, scales, zero points
record.io.state_pairs              # explicit state: which output feeds which input

model = record.fetch("a8w8")       # a local path whose sha256 matches the record
golden = record.golden("a8w8")     # golden.inputs / golden.outputs as NumPy arrays
zoo.resolve("zoo://rnnoise/a8w8")  # the same model by URI
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
- `hf://` sources (`hf://[datasets/]<org>/<repo>@<40-hex commit>/<path>`) need
  the `hf` extra and use huggingface_hub's own login (`HF_TOKEN` or
  `hf auth login`).
- Every model, golden or other file with a sha256 is checked for size and
  sha256 before it enters the cache (`HELIA_ZOO_CACHE`, default
  `~/.cache/helia-model-zoo`), and checked again on every fetch. A card or
  overlay record without a sha256 is pinned by its revision instead.

### Private records

Private models never enter this repository. Their records live in an overlay
with this repository's `models/<id>/record.json` layout that only its users can
read. `HELIA_ZOO_OVERLAY` names it: a local directory, or a Hugging Face dataset
root such as `hf://datasets/<org>/<repo>@<40-hex commit>`, read once per
process. Its records join the packaged ones. Before pushing any change here, run

```bash
helia-zoo guard --overlay "$HELIA_ZOO_OVERLAY" --text pr-body.md
```

It refuses if any overlay ID, title, Hugging Face repository, upstream
or sha256 appears in what a push would publish: the content, paths, messages
and authors of every commit since `origin/main`, the index and working tree
(including file names and symlink targets), the branch name, or the extra text
files. Names that the records at `origin/main` also use are public and are not
reported.
