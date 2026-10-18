# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import json
import shutil

import pytest
from conftest import REPO

from helia_model_zoo.v1 import validate_manifest

V1 = {item["id"]: item for item in json.loads((REPO / "corpus-manifest-v1.json").read_text())["entries"]}


def manifest(root, tmp_path, *items):
    """A v1 manifest of ``items`` over copies of the files they name."""
    for item in items:
        for key in ("model", "golden", "provenance_reference", "license_reference"):
            (tmp_path / item[key]).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / item[key], tmp_path / item[key])
    path = tmp_path / "corpus-manifest-v1.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": list(items)}))
    return path


def golden_of(item_id):
    return {key: V1[item_id][key] for key in ("golden", "golden_sha256")}


def test_a_float16_model_takes_its_own_records_fp32_golden(root, tmp_path):
    validate_manifest(manifest(root, tmp_path, V1["mlperf-tiny-kws-fp16"], V1["mlperf-tiny-resnet8-fp16"]))


@pytest.mark.parametrize("other", ["mlperf-tiny-resnet8-fp32", "mlperf-tiny-kws-int16", "mlperf-tiny-kws-int8"])
def test_a_float16_model_refuses_any_other_golden(root, tmp_path, other):
    item = {**V1["mlperf-tiny-kws-fp16"], **golden_of(other)}
    with pytest.raises(ValueError, match="must name its record's fp32 golden models/mlperf-tiny-kws/fp32/golden.npz"):
        validate_manifest(manifest(root, tmp_path, item))


@pytest.mark.parametrize(
    ("model", "golden", "dtype"),
    [
        ("mlperf-tiny-kws-fp32", "mlperf-tiny-kws-int16", "expected float32, got int16"),
        ("mlperf-tiny-kws-int16", "mlperf-tiny-kws-fp32", "expected int16, got float32"),
        ("mlperf-tiny-kws-int8", "mlperf-tiny-kws-fp32", "expected int8, got float32"),
    ],
)
def test_only_a_float16_model_takes_a_float32_golden_of_another_dtype(root, tmp_path, model, golden, dtype):
    item = {**V1[model], **golden_of(golden)}
    with pytest.raises(ValueError, match=f"input_0 dtype mismatch: {dtype}"):
        validate_manifest(manifest(root, tmp_path, item))
