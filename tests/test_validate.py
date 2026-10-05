# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import json
import os
import shutil
import subprocess
from importlib import metadata

import numpy as np
import pytest
from conftest import POINTER, REPO, entry

from helia_model_zoo.cli import main
from helia_model_zoo.manifest import parse_records
from helia_model_zoo.validate import ValidationError, validate


def problems_of(root, data, **options):
    with pytest.raises(ValidationError) as caught:
        validate(root, parse_records(data), **options)
    return caught.value.problems


def only(data, *model_ids):
    for model_id in [m for m in data if m not in model_ids]:
        del data[model_id]
    return data


def copy_entry(root, tmp_path, data, model_id):
    """Copy one record's files into tmp_path, so tests can damage them."""
    item = entry(data, model_id)
    paths = [item["card"]]
    for precision in item["precisions"].values():
        paths += [precision["model"]["path"]] + ([precision["golden"]["path"]] if "golden" in precision else [])
    for path in paths:
        relative = f"models/{model_id}/{path}"
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, tmp_path / relative)
    return only(data, model_id)


def test_packaged_manifest_validates_with_replay_and_v1(root):
    validate(root, replay=True, v1=root / "corpus-manifest-v1.json")


def test_private_entry_is_refused_before_any_file_is_read(tmp_path, data):
    entry(data, "rnnoise")["visibility"] = "private"
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False)
    assert problems == ["record rnnoise: visibility is 'private'; this repository holds public records only"]


def test_lfs_pointer_is_refused(tmp_path, data):
    item = entry(data, "rnnoise")
    for relative in (item["card"], item["precisions"]["a8w8"]["model"]["path"]):
        path = tmp_path / "models/rnnoise" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(POINTER + b"\noid sha256:0\n")
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False)
    assert (
        "record rnnoise.precisions.a8w8.model: unresolved Git LFS pointer lfs://models/rnnoise/a8w8/model.tflite"
        in problems
    )


def test_missing_files_are_all_reported(tmp_path, data):
    problems = problems_of(tmp_path, only(data, "rnnoise", "wav2letter"), signatures=False)
    assert sum("missing file" in p for p in problems) == 6


def test_changed_model_byte_is_refused(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-kws")
    model = tmp_path / "models/mlperf-tiny-kws/a8w8/model.tflite"
    content = bytearray(model.read_bytes())
    content[-1] ^= 1
    model.write_bytes(bytes(content))
    problems = problems_of(tmp_path, data)
    assert len(problems) == 1 and "sha256 mismatch for lfs://models/mlperf-tiny-kws/a8w8/model.tflite" in problems[0]


def test_changed_model_size_is_refused(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-kws")
    with (tmp_path / "models/mlperf-tiny-kws/a8w8/model.tflite").open("ab") as model:
        model.write(b"\0")
    expected = entry(data, "mlperf-tiny-kws")["precisions"]["a8w8"]["model"]["bytes"]
    assert problems_of(tmp_path, data) == [
        f"record mlperf-tiny-kws.precisions.a8w8.model: lfs://models/mlperf-tiny-kws/a8w8/model.tflite is "
        f"{expected + 1} bytes, expected {expected}"
    ]


def test_declared_tensor_must_match_the_model(root, data):
    entry(data, "rnnoise")["precisions"]["a8w8"]["inputs"][1]["scale"] = 0.0078431377
    problems = problems_of(root, only(data, "rnnoise"))
    assert any(p.startswith("record rnnoise.precisions.a8w8.inputs[1]: record") for p in problems)


@pytest.mark.parametrize(("model_id", "resolver"), [("mlperf-tiny-resnet-wide", "builtin"), ("rnnoise", "builtin")])
def test_replay_under_the_wrong_resolver_is_refused(root, data, model_id, resolver):
    entry(data, model_id)["precisions"]["a8w8"]["golden"]["resolver"] = resolver
    only(data, model_id)
    validate(root, parse_records(data))
    problems = problems_of(root, data, replay=True)
    assert problems and all("does not replay exactly" in p for p in problems)


def test_changed_golden_output_fails_replay_only(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-vww")
    golden = tmp_path / "models/mlperf-tiny-vww/a8w8/golden.npz"
    arrays = dict(np.load(golden))
    arrays["output_0"] = arrays["output_0"].copy()
    arrays["output_0"].flat[0] += 1
    np.savez(golden, **arrays)
    precision = entry(data, "mlperf-tiny-vww")["precisions"]["a8w8"]
    precision["golden"].update(sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size)
    validate(tmp_path, parse_records(data))
    problems = problems_of(tmp_path, data, replay=True)
    assert problems == [
        "record mlperf-tiny-vww.precisions.a8w8.golden: output_0 does not replay exactly under "
        "builtin_ref (max abs difference 1)"
    ]


def test_golden_keys_and_dtypes_are_checked(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-ad01")
    golden = tmp_path / "models/mlperf-tiny-ad01/a8w8/golden.npz"
    arrays = dict(np.load(golden))
    arrays["input_0"] = arrays["input_0"].astype(np.int16)
    arrays["output_1"] = arrays["output_0"]
    np.savez(golden, **arrays)
    entry(data, "mlperf-tiny-ad01")["precisions"]["a8w8"]["golden"].update(
        sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size
    )
    problems = problems_of(tmp_path, data)
    assert len(problems) == 1 and "expected keys ['input_0', 'output_0']" in problems[0]
    del arrays["output_1"]
    np.savez(golden, **arrays)
    entry(data, "mlperf-tiny-ad01")["precisions"]["a8w8"]["golden"].update(
        sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size
    )
    assert problems_of(tmp_path, data) == [
        "record mlperf-tiny-ad01.precisions.a8w8.golden: input_0 dtype int16, expected int8"
    ]


V1_FIELDS = {
    "model_sha256": "0" * 64,
    "golden": "audio/x/golden.npz",
    "golden_sha256": "1" * 64,
    "reference_runtime": "tflite-runtime",
    "reference_runtime_version": "9.9.9",
    "provenance_reference": "audio/x/README.md",
    "license_reference": "audio/x/LICENSE.md",
}


@pytest.mark.parametrize("field", sorted(V1_FIELDS))
def test_v1_and_records_must_agree_on_every_field(tmp_path, data, field):
    v1 = json.loads((REPO / "corpus-manifest-v1.json").read_text())
    next(i for i in v1["entries"] if i["id"] == "rnnoise-int8")[field] = V1_FIELDS[field]
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1))
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False, v1=path)
    v1_problems = [p for p in problems if p.startswith("v1 rnnoise-int8")]
    assert len(v1_problems) == 1 and v1_problems[0].startswith(f"v1 rnnoise-int8: {field} is ")
    assert V1_FIELDS[field] in v1_problems[0]


def test_v1_ids_must_name_the_record(tmp_path, data):
    v1 = json.loads((REPO / "corpus-manifest-v1.json").read_text())
    item = next(i for i in v1["entries"] if i["id"] == "rnnoise-int8")
    path = tmp_path / "v1.json"
    for wrong in ("wav2letter-int8", "rnnoise", "rnnoise-"):
        item["id"] = wrong
        path.write_text(json.dumps(v1))
        problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False, v1=path)
        assert f"v1 {wrong}: the ID does not name record rnnoise, which lists its model" in problems


def test_v1_models_must_be_in_a_record(tmp_path, data):
    v1 = json.loads((REPO / "corpus-manifest-v1.json").read_text())
    v1["entries"].append(dict(v1["entries"][1], id="not-in-a-record", model="models/x/a8w8/model.tflite"))
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1))
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False, v1=path)
    assert problems[-1] == "v1 not-in-a-record: no record lists the model 'models/x/a8w8/model.tflite'"


def write_golden(path, arrays, precision):
    np.savez(path, **arrays)
    precision["golden"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def test_golden_shape_and_missing_key_are_checked(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-ad01")
    golden = tmp_path / "models/mlperf-tiny-ad01/a8w8/golden.npz"
    precision = entry(data, "mlperf-tiny-ad01")["precisions"]["a8w8"]
    arrays = dict(np.load(golden))
    write_golden(golden, {**arrays, "output_0": arrays["output_0"].reshape(640, 1)}, precision)
    assert problems_of(tmp_path, data) == [
        "record mlperf-tiny-ad01.precisions.a8w8.golden: output_0 shape (640, 1), expected (1, 640)"
    ]
    write_golden(golden, {"input_0": arrays["input_0"]}, precision)
    assert problems_of(tmp_path, data) == [
        "record mlperf-tiny-ad01.precisions.a8w8.golden: expected keys ['input_0', 'output_0'], found ['input_0']"
    ]


def test_batch_golden_needs_the_step_axis(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-ad01")
    golden = tmp_path / "models/mlperf-tiny-ad01/a8w8/golden.npz"
    precision = entry(data, "mlperf-tiny-ad01")["precisions"]["a8w8"]
    precision["golden"].update(kind="batch", steps=2)
    arrays = dict(np.load(golden))
    write_golden(golden, {k: np.stack([v, v]) for k, v in arrays.items()}, precision)
    validate(tmp_path, parse_records(data))
    validate(tmp_path, parse_records(data), replay=True)
    write_golden(golden, arrays, precision)
    assert problems_of(tmp_path, data) == [
        "record mlperf-tiny-ad01.precisions.a8w8.golden: input_0 shape (1, 640), expected (2, 1, 640)",
        "record mlperf-tiny-ad01.precisions.a8w8.golden: output_0 shape (1, 640), expected (2, 1, 640)",
    ]


def _state_problems(data, change):
    item = entry(data, "rnnoise")
    change(item["precisions"]["a8w8"])
    with pytest.raises(ValidationError) as caught:
        validate(REPO, parse_records(only(data, "rnnoise")), signatures=False)
    return [p for p in caught.value.problems if "state pair" in p]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p: p["inputs"][3].update(shape=[1, 95]),
            "state pair 2 (denoise_gru_prev_state_int8 <- Identity_int8): element counts differ",
        ),
        (
            lambda p: p["outputs"][2].update(dtype="int16"),
            "state pair 1 (noise_gru_prev_state_int8 <- Identity_2_int8): dtype or zero point differs",
        ),
        (
            lambda p: p["inputs"][1].update(zero_point=0),
            "state pair 0 (vad_gru_prev_state_int8 <- Identity_3_int8): dtype or zero point differs",
        ),
    ],
)
def test_state_pair_tensors_must_agree(data, change, message):
    assert _state_problems(data, change) == [f"record rnnoise.precisions.a8w8: {message}"]


def test_guard_refuses_any_private_entry(tmp_path, data):
    entry(data, "wav2letter")["visibility"] = "private"
    problems = problems_of(tmp_path, only(data, "rnnoise", "wav2letter"), signatures=False)
    assert "record wav2letter: visibility is 'private'; this repository holds public records only" in problems


def write_records(directory, data):
    for model_id, record in data.items():
        (directory / model_id).mkdir(parents=True)
        (directory / model_id / "record.json").write_text(json.dumps(record))
    return directory


def test_cli_refuses_a_private_record(root, tmp_path, data, capsys):
    entry(data, "mobilenet-v2")["visibility"] = "private"
    models = write_records(tmp_path / "models", data)
    assert main(["--models", str(models), "validate", "--root", str(root)]) == 1
    assert "record mobilenet-v2: visibility is 'private'" in capsys.readouterr().err


def test_replay_checks_every_output(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "rnnoise")
    golden = tmp_path / "models/rnnoise/a8w8/golden.npz"
    precision = entry(data, "rnnoise")["precisions"]["a8w8"]
    arrays = dict(np.load(golden))
    arrays["output_4"] = arrays["output_4"] + np.int8(1)
    write_golden(golden, arrays, precision)
    assert problems_of(tmp_path, data, replay=True) == [
        "record rnnoise.precisions.a8w8.golden: output_4 does not replay exactly under builtin_ref (max abs difference 1)"
    ]


def test_replay_is_skipped_after_a_signature_mismatch(root, tmp_path, data):
    # The golden stays consistent with the manifest, and its output would fail replay.
    copy_entry(root, tmp_path, data, "mlperf-tiny-vww")
    golden = tmp_path / "models/mlperf-tiny-vww/a8w8/golden.npz"
    precision = entry(data, "mlperf-tiny-vww")["precisions"]["a8w8"]
    arrays = dict(np.load(golden))
    arrays["output_0"] = arrays["output_0"] + np.int8(1)
    write_golden(golden, arrays, precision)
    precision["inputs"][0]["scale"] = 0.5
    problems = problems_of(tmp_path, data, replay=True)
    assert len(problems) == 1 and problems[0].startswith("record mlperf-tiny-vww.precisions.a8w8.inputs[0]: record")


def test_replay_needs_the_recorded_runtime_version(root, data):
    entry(data, "mlperf-tiny-kws")["precisions"]["a8w8"]["golden"]["runtime"] = "ai-edge-litert==0.0.1"
    problems = problems_of(root, only(data, "mlperf-tiny-kws"), replay=True)
    installed = metadata.version("ai-edge-litert")
    assert problems == [
        f"record mlperf-tiny-kws.precisions.a8w8.golden: replay needs ai-edge-litert 0.0.1; installed: {installed}"
    ]


def test_symlink_escape_is_reported(tmp_path, data):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "README.md").write_text("card")
    (tmp_path / "root/models").mkdir(parents=True)
    (tmp_path / "root/models/rnnoise").symlink_to(outside)
    problems = problems_of(tmp_path / "root", only(data, "rnnoise"), signatures=False)
    assert "record rnnoise.card: repo://models/rnnoise/README.md escapes the repository root" in problems


def test_cli(root, tmp_path, capsys):
    assert main(["list"]) == 0
    assert "rnnoise\ta8w8\tspeech-denoising\tpublic\tRNNoise" in capsys.readouterr().out
    assert main(["show", "dfnet2"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "dfnet2"
    assert main(["show", "nope"]) == 1
    assert main(["validate", "--root", str(root), "--replay"]) == 0
    assert main(["validate", "--root", str(tmp_path), "--no-signatures"]) == 2
    assert main(["validate", "--root", str(root), "--replay", "--no-signatures"]) == 2
    assert main(["validate", "--root", str(tmp_path), "--no-signatures", "--no-v1"]) == 2
    assert "no records under" in capsys.readouterr().err
    models = str(REPO / "models")
    assert main(["--models", models, "validate", "--root", str(tmp_path), "--no-signatures", "--no-v1"]) == 1
    assert "problem(s)" in capsys.readouterr().err


def test_cli_without_litert_explains_the_extra(root, monkeypatch, capsys):
    def missing():
        raise ImportError("signature and replay checks need LiteRT: install helia-model-zoo[litert]")

    monkeypatch.setattr("helia_model_zoo.runtime.litert_module", missing)
    assert main(["validate", "--root", str(root)]) == 2
    assert "install helia-model-zoo[litert]" in capsys.readouterr().err
    assert main(["validate", "--root", str(root), "--no-signatures"]) == 0


def track(root, *paths):
    """Make ``root`` a git checkout tracking ``paths`` (default: everything under it)."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    subprocess.run(["git", "init", "-q", str(root)], check=True, env=environment)
    subprocess.run(["git", "-C", str(root), "add", "-A", "--", *(paths or ["."])], check=True, env=environment)


def test_a_complete_check_refuses_tracked_unlisted_artifacts(root, tmp_path, data, monkeypatch):
    data = copy_entry(root, tmp_path / "outer/inner", data, "rnnoise")
    (tmp_path / "outer/README.md").write_text("outer")
    track(tmp_path / "outer", "README.md")
    assert problems_of(tmp_path / "outer/inner", data, signatures=False, complete=True) == [
        f"{tmp_path / 'outer/inner'}: not the top of its git checkout ({tmp_path / 'outer'})"
    ]
    tmp_path = tmp_path / "outer/inner"
    for stray in ("audio/x/MODEL.TFLITE", "models/rnnoise/extra.npz", "venv/lib/data.npz", "models/rnnoise/notes.txt"):
        (tmp_path / stray).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / stray).write_bytes(b"x")
    (tmp_path / "models/rnnoise/loop.tflite").symlink_to("loop.tflite")
    track(tmp_path, "models", "audio")
    validate(tmp_path, parse_records(data), signatures=False)
    expected = [
        "audio/x/MODEL.TFLITE: no record lists this artifact",
        "models/rnnoise/extra.npz: no record lists this artifact",
        "models/rnnoise/loop.tflite: no record lists this artifact",
    ]
    assert problems_of(tmp_path, data, signatures=False, complete=True) == expected
    # Inside a git hook, GIT_DIR and GIT_INDEX_FILE name the hook's repository.
    outer = tmp_path.parent
    monkeypatch.setenv("GIT_DIR", str(outer / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(outer / ".git/index"))
    assert problems_of(tmp_path, data, signatures=False, complete=True) == expected


def test_cli_validate_refuses_unlisted_artifacts_in_its_root(root, tmp_path, data, capsys):
    item = entry(copy_entry(root, tmp_path, data, "rnnoise"), "rnnoise")
    (tmp_path / "models/rnnoise/record.json").write_text(json.dumps(item))
    (tmp_path / "models/rnnoise/a8w8/untracked.tflite").write_bytes(b"x")
    track(tmp_path, "models/rnnoise/record.json", "models/rnnoise/README.md", "models/rnnoise/a8w8/model.tflite")
    command = ["validate", "--root", str(tmp_path), "--no-signatures", "--no-v1"]
    assert main(command) == 0
    track(tmp_path)
    assert main(command) == 1
    assert "models/rnnoise/a8w8/untracked.tflite: no record lists this artifact" in capsys.readouterr().err
    assert main(["--models", str(tmp_path / "models"), *command]) == 0


def test_cli_inventory(tmp_path, data, capsys):
    assert main(["list", "--json"]) == 0
    rows = {row["id"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows["dfnet2"] | {"title": None} == {
        "id": "dfnet2",
        "title": None,
        "task": "speech-enhancement",
        "domain": "audio",
        "source": "https://github.com/Rikorose/DeepFilterNet",
        "license": None,
        "redistributable": "unverified",
        "precisions": ["a16w8"],
        "golden": ["a16w8"],
        "hosting": ["git-lfs"],
        "gaps": ["licence unknown", "source revision unknown"],
    }
    vww = rows["mlperf-tiny-vww"]
    assert (vww["precisions"], vww["golden"], vww["redistributable"], vww["gaps"]) == (
        ["fp32", "fp16", "a8w8"],
        ["fp32", "a8w8"],
        "yes",
        ["no fp16 golden"],
    )
    assert vww["source"] == "https://github.com/mlcommons/tiny@4addd0fa08d216e20637637874e084895f289da4"
    with pytest.raises(SystemExit):
        main(["list", "--json", "--markdown"])
    capsys.readouterr()

    item = entry(data, "rnnoise")
    item.update(task="de|noise\nfilter", license={"spdx": "NOASSERTION"})
    wide = json.loads(json.dumps(item["precisions"]["a8w8"]))
    wide["golden"]["uri"] = "hf://datasets/Example/goldens@" + "b" * 40 + "/fp32.npz"
    del wide["golden"]["path"]
    item["precisions"]["fp32"] = wide
    precision = item["precisions"]["a8w8"]
    precision.pop("golden")
    precision["model"].pop("path")
    precision["model"]["uri"] = "hf://Example/rnnoise@" + "a" * 40 + "/model.tflite"
    secret = json.loads(json.dumps(entry(data, "mlperf-tiny-ad01")))
    secret.update(id="secret", visibility="private")
    for model_id, record in (("rnnoise", item), ("secret", secret)):
        (tmp_path / "models" / model_id).mkdir(parents=True)
        (tmp_path / "models" / model_id / "record.json").write_text(json.dumps(record))
    assert main(["--models", str(tmp_path / "models"), "list", "--markdown"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "| ID | Task | Source | Licence | Redistributable | Precisions | Golden | Hosting | Gaps |"
    assert lines[2:] == [
        "| rnnoise | de\\|noise filter | ARM-software/ML-zoo@fec0bb5b | NOASSERTION | unverified | fp32, a8w8 | fp32 "
        "| git-lfs, hf:Example/rnnoise, hf:datasets/Example/goldens | licence unknown, no a8w8 golden |",
        "| secret | anomaly-detection | mlcommons/tiny@4addd0fa | Apache-2.0 | private | fp32, fp16, a8w8 "
        "| fp32, a8w8 | git-lfs | no fp16 golden |",
    ]


def test_a_float_precision_must_hold_a_float_model(root, data):
    item = entry(data, "rnnoise")
    item["precisions"]["fp32"] = item["precisions"].pop("a8w8")
    problems = problems_of(root, only(data, "rnnoise"), replay=True)
    assert "record rnnoise.precisions.fp32: not a float model: 94 of its tensors are int8, uint8 or int16" in problems


def test_an_fp16_precision_must_hold_a_native_float16_model(root, data):
    item = entry(data, "mlperf-tiny-kws")
    item["precisions"]["fp16"] = item["precisions"].pop("fp32")
    problems = problems_of(root, only(data, "mlperf-tiny-kws"), signatures=True)
    assert any("precisions.fp16: not a native float16 model:" in p for p in problems)


def test_golden_generate_explains_a_model_litert_cannot_run(root, tmp_path, capsys):
    model = root / "models/mlperf-tiny-kws/fp16/model.tflite"
    assert main(["golden", "generate", str(model), str(tmp_path / "g.npz")]) == 1
    assert f"LiteRT cannot run {model}" in capsys.readouterr().err
