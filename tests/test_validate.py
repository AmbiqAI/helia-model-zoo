# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import json
import shutil

import numpy as np
import pytest
from conftest import POINTER, entry

from helia_model_zoo.cli import main
from helia_model_zoo.manifest import parse_manifest
from helia_model_zoo.validate import ValidationError, validate


def problems_of(root, data, **options):
    with pytest.raises(ValidationError) as caught:
        validate(root, parse_manifest(data), **options)
    return caught.value.problems


def only(data, *model_ids):
    data["entries"] = [entry(data, m) for m in model_ids]
    return data


def copy_entry(root, tmp_path, data, model_id):
    """Copy one entry's files into tmp_path, so tests can damage them."""
    item = entry(data, model_id)
    refs = [item["card"]["uri"], item["license"]["reference"]["uri"]]
    for precision in item["precisions"].values():
        refs += [precision["model"]["uri"], precision["golden"]["file"]["uri"]]
    for uri in refs:
        relative = uri.split("://", 1)[1]
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, tmp_path / relative)
    return only(data, model_id)


def test_packaged_manifest_validates_with_replay_and_v1(root):
    validate(root, replay=True, v1=root / "corpus-manifest-v1.json")


def test_private_entry_is_refused_before_any_file_is_read(tmp_path, data):
    entry(data, "rnnoise")["visibility"] = "private"
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False)
    assert problems == ["entry rnnoise: visibility is 'private'; this manifest holds public entries only"]


def test_lfs_pointer_is_refused(tmp_path, data):
    item = entry(data, "rnnoise")
    for uri in (item["card"]["uri"], item["precisions"]["int8"]["model"]["uri"]):
        path = tmp_path / uri.split("://", 1)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(POINTER + b"\noid sha256:0\n")
    problems = problems_of(tmp_path, only(data, "rnnoise"), signatures=False)
    assert (
        "entry rnnoise.precisions.int8.model: unresolved Git LFS pointer lfs://audio/rnnoise/model.tflite" in problems
    )


def test_missing_files_are_all_reported(tmp_path, data):
    problems = problems_of(tmp_path, only(data, "rnnoise", "wav2letter"), signatures=False)
    assert sum("missing file" in p for p in problems) == 8


def test_changed_model_byte_is_refused(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-kws")
    model = tmp_path / "audio/mlperf-tiny/kws_ref/model.tflite"
    content = bytearray(model.read_bytes())
    content[-1] ^= 1
    model.write_bytes(bytes(content))
    problems = problems_of(tmp_path, data)
    assert len(problems) == 1 and "sha256 mismatch for lfs://audio/mlperf-tiny/kws_ref/model.tflite" in problems[0]


def test_changed_model_size_is_refused(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-kws")
    with (tmp_path / "audio/mlperf-tiny/kws_ref/model.tflite").open("ab") as model:
        model.write(b"\0")
    expected = entry(data, "mlperf-tiny-kws")["precisions"]["int8"]["model"]["bytes"]
    assert problems_of(tmp_path, data) == [
        f"entry mlperf-tiny-kws.precisions.int8.model: lfs://audio/mlperf-tiny/kws_ref/model.tflite is "
        f"{expected + 1} bytes, expected {expected}"
    ]


def test_declared_tensor_must_match_the_model(root, data):
    entry(data, "rnnoise")["precisions"]["int8"]["inputs"][1]["scale"] = 0.0078431377
    problems = problems_of(root, only(data, "rnnoise"))
    assert any(p.startswith("entry rnnoise.precisions.int8.inputs[1]: manifest") for p in problems)


def test_scales_tied_must_be_true_to_the_model(root, data):
    entry(data, "dfnet2")["io"]["state_pairs"][0]["scales_tied"] = False
    problems = problems_of(root, only(data, "dfnet2"))
    assert len(problems) == 1 and "scales_tied is False" in problems[0]


@pytest.mark.parametrize(("model_id", "resolver"), [("mlperf-tiny-resnet", "builtin_ref"), ("rnnoise", "builtin")])
def test_replay_under_the_wrong_resolver_is_refused(root, data, model_id, resolver):
    entry(data, model_id)["precisions"]["int8"]["golden"]["resolver"] = resolver
    only(data, model_id)
    validate(root, parse_manifest(data))
    problems = problems_of(root, data, replay=True)
    assert problems and all("does not replay exactly" in p for p in problems)


def test_changed_golden_output_fails_replay_only(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-vww")
    golden = tmp_path / "vision/mlperf-tiny/vww/golden.npz"
    arrays = dict(np.load(golden))
    arrays["output_0"] = arrays["output_0"].copy()
    arrays["output_0"].flat[0] += 1
    np.savez(golden, **arrays)
    precision = entry(data, "mlperf-tiny-vww")["precisions"]["int8"]
    precision["golden"]["file"].update(
        sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size
    )
    validate(tmp_path, parse_manifest(data))
    problems = problems_of(tmp_path, data, replay=True)
    assert problems == [
        "entry mlperf-tiny-vww.precisions.int8.golden: output_0 does not replay exactly under "
        "builtin_ref (max abs difference 1)"
    ]


def test_golden_keys_and_dtypes_are_checked(root, tmp_path, data):
    copy_entry(root, tmp_path, data, "mlperf-tiny-ad01")
    golden = tmp_path / "anomaly-detection/mlperf-tiny/ad01/golden.npz"
    arrays = dict(np.load(golden))
    arrays["input_0"] = arrays["input_0"].astype(np.int16)
    arrays["output_1"] = arrays["output_0"]
    np.savez(golden, **arrays)
    entry(data, "mlperf-tiny-ad01")["precisions"]["int8"]["golden"]["file"].update(
        sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size
    )
    problems = problems_of(tmp_path, data)
    assert len(problems) == 1 and "expected keys ['input_0', 'output_0']" in problems[0]
    del arrays["output_1"]
    np.savez(golden, **arrays)
    entry(data, "mlperf-tiny-ad01")["precisions"]["int8"]["golden"]["file"].update(
        sha256=hashlib.sha256(golden.read_bytes()).hexdigest(), bytes=golden.stat().st_size
    )
    assert problems_of(tmp_path, data) == [
        "entry mlperf-tiny-ad01.precisions.int8.golden: input_0 dtype int16, expected int8"
    ]


def test_v1_and_v2_must_agree(root, tmp_path, data):
    v1 = json.loads((root / "corpus-manifest-v1.json").read_text())
    v1["entries"][0]["golden_sha256"] = "0" * 64
    v1["entries"].append(dict(v1["entries"][1], id="not-in-v2"))
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1))
    problems = problems_of(root, data, v1=path)
    assert problems == [
        f"v1 {v1['entries'][0]['id']}: golden_sha256 is '{'0' * 64}' in v1 but "
        f"'{entry(data, 'mlperf-tiny-ad01')['precisions']['int8']['golden']['file']['sha256']}' in v2",
        "v1 not-in-v2: no v2 entry has this alias",
    ]


def test_cli(root, tmp_path, capsys):
    assert main(["list"]) == 0
    assert "rnnoise\tint8\tconverted\tpublic\tRNNoise" in capsys.readouterr().out
    assert main(["show", "dfnet2-int16"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "dfnet2"
    assert main(["show", "nope"]) == 1
    assert main(["validate", "--root", str(root), "--replay"]) == 0
    assert main(["validate", "--root", str(tmp_path), "--no-signatures"]) == 2
    assert main(["validate", "--root", str(root), "--replay", "--no-signatures"]) == 2
    assert main(["validate", "--root", str(tmp_path), "--no-signatures", "--no-v1"]) == 1
    assert "problem(s)" in capsys.readouterr().err
