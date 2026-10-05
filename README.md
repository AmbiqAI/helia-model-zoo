# heliaAOT Model Zoo

This repository is a model zoo for prebuilt TFLite models. Each model has one
small record, its card, and per precision its TFLite artifact and, where LiteRT can
run it, a golden fixture (LiteRT has no kernels for native float16 graphs).

## Repository layout

```text
models/<id>/
  record.json          helia-model-zoo/record@1
  README.md            the model card
  <precision>/model.tflite
  <precision>/golden.npz      (every precision except fp16)
```

The directory name is the model ID. Precisions are `fp32`, `fp16`, `a8w8`,
`a16w8` and `a8w4`. There is no other inventory: `helia-zoo list --markdown`
(or `--json`) prints one from the records. For each model it gives the task,
source, licence, whether it is redistributable, its precisions, which of them
have a golden, where its files are hosted, and its known gaps. Redistributable
is `yes` for a public record with a licence (whose terms still apply),
`unverified` for one without, and `private` for an overlay record.

`helia-zoo validate` refuses a golden on an `fp16` precision, which LiteRT
cannot replay. Its signature checks, skipped with `--no-signatures`, also
refuse any `fp32` or `fp16` model that holds int8, uint8 or int16 tensors, any
`fp16` model that holds float32 or float64 tensors, and float16 tensors in any
other precision. An `fp16` precision without a golden is not a gap. On this
repository's own records, it also refuses any tracked `.tflite` or `.npz` file
that no record lists; `--root` must then be the top of a git checkout. A record
with several precisions needs the precision named (`zoo://<id>/<precision>`).

To contribute a model/golden pair for helia-aot release testing, follow
[Add a model to the release corpus](docs/how-to/add-release-model.md). Adding an
artifact here and enabling it in helia-aot are separate, reviewed changes.

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
upstream source. The records ship with the Python package.
`corpus-manifest-v1.json` keeps the v1 format that helia-aot reads; CI checks
that each v1 entry agrees with the record holding its model, and that its ID is
that record's ID with a suffix.

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
python -m pip install --no-deps -e .
helia-zoo golden generate path/to/model.tflite path/to/golden.npz --seed 42
python tools/validate_corpus.py corpus-manifest-v1.json
```

`helia-zoo golden generate` writes a `single` golden with LiteRT's reference
kernels by default. For many models their outputs differ from the optimized
kernels', which `--resolver builtin` selects. A model with `state_in_k`/`state_out_k` state
names gets its state inputs at the reset value, not drawn from the seed.

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

art = zoo.resolve("zoo://rnnoise/a8w8@<40-hex commit>")  # the model as recorded at that commit
art.record, art.precision          # the record and precision at that commit
art.model, art.golden              # their verified local files
```

A reference is `zoo://<id>[/<precision>][@<commit>]`; `zoo.parse_reference()`
splits one. With a commit, the record is downloaded from this repository at that
commit (`zoo.get(id, revision)`), and its files come from the same commit. This
works for any commit from 83ed7226 on whose records use the installed version's
schema (`record@1`). An ID is private only when `HELIA_ZOO_OVERLAY` holds it; it
then resolves only at the revision the overlay is pinned to, and any other ID is
looked up here. Without a commit, the installed record is used, so pin one
wherever results must be reproducible.

`helia-zoo list [--json | --markdown]`, `show <id>`,
`fetch <id> [--precision P] [--golden] [--card] [--root DIR]` and `validate`
expose the same on the command line. Installing from Git with git-lfs present
downloads every artifact in the repository; set `GIT_LFS_SKIP_SMUDGE=1` to skip
them.

Files are fetched as follows:

- Files in this repository are read from a hydrated checkout when there is one
  (`root=`, `HELIA_ZOO_ROOT`, or an editable install). Otherwise they are
  downloaded from GitHub at the commit the package was installed from
  (`HELIA_ZOO_REVISION` overrides it). A checkout cannot show its commit, so at
  an explicit revision only a `root=` checkout is read, and only for files with
  a sha256; a record or card is downloaded once at that revision, then cached.
- `hf://` sources (`hf://[datasets/]<org>/<repo>@<40-hex commit>/<path>`) need
  the `hf` extra and use huggingface_hub's own login (`HF_TOKEN` or
  `hf auth login`).
- Every model, golden or other file with a sha256 is checked for size and
  sha256 before it enters the cache (`HELIA_ZOO_CACHE`, default
  `~/.cache/helia-model-zoo`), and checked again on every fetch. A card, a record
  read at a revision, or an overlay record has no sha256 and is pinned by its
  revision instead.

### Private records

Private models never enter this repository. Their records live in an overlay
with this repository's `models/<id>/record.json` layout that only its users can
read. `HELIA_ZOO_OVERLAY` names it: a local directory, or a Hugging Face dataset
root such as `hf://datasets/<org>/<repo>@<40-hex commit>`, read once per
process. Its records join the packaged ones, and give every file as a pinned
`hf://` URI rather than a path. Before pushing any change here, run

```bash
helia-zoo guard --overlay "$HELIA_ZOO_OVERLAY" --text pr-body.md
```

It refuses if any overlay ID, title, Hugging Face repository, upstream
or sha256 appears in what a push would publish: the content, paths, messages
and authors of every commit since `origin/main`, the index and working tree
(including file names and symlink targets), the branch name, or the extra text
files. Names that the records at `origin/main` also use are public and are not
reported.
