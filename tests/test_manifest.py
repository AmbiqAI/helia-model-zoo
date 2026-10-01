# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import json

import pytest
from conftest import PACKAGED, REPO, entry

import helia_model_zoo as zoo
from helia_model_zoo.manifest import ManifestError, parse_manifest


def test_packaged_manifest_lists_the_v1_corpus():
    v1 = json.loads((REPO / "corpus-manifest-v1.json").read_text())
    assert len(zoo.entries()) == len(v1["entries"]) == 9
    for item in v1["entries"]:
        _, precision = zoo.manifest().resolve(item["id"])
        assert precision is not None and precision.model.sha256 == item["model_sha256"]
    assert all(e.visibility == "public" for e in zoo.entries())


def test_get_by_id_and_alias():
    assert zoo.get("rnnoise") is zoo.get("rnnoise-int8")
    assert zoo.manifest().resolve("rnnoise") == (zoo.get("rnnoise"), None)
    assert zoo.manifest().resolve("dfnet2-int16")[1].name == "int16x8"
    with pytest.raises(KeyError):
        zoo.get("no-such-model")


def test_state_pairs_are_recorded():
    rnnoise = zoo.get("rnnoise")
    assert rnnoise.io.streaming == "explicit_state"
    assert [(p.input, p.output, p.scales_tied) for p in rnnoise.io.state_pairs] == [
        (1, 3, False),
        (2, 2, False),
        (3, 0, False),
    ]
    assert [(p.input, p.output, p.scales_tied) for p in zoo.get("dfnet2").io.state_pairs] == [(2, 4, True)]
    assert zoo.get("mlperf-tiny-kws").io.state_pairs == ()


GOLDEN = lambda e: e["precisions"]["int8"]["golden"]  # noqa: E731
CASES = {
    "schema": (None, lambda d: d.update(schema="helia-model-zoo/manifest@1"), "unsupported schema"),
    "unknown key": ("rnnoise", lambda e: e.update(extra=1), "unknown keys"),
    "duplicate id": (None, lambda d: d["entries"].append(dict(entry(d, "rnnoise"))), "already used"),
    "alias collides with an id": ("rnnoise", lambda e: e["aliases"].update({"wav2letter": "int8"}), "already used"),
    "alias names a missing precision": (
        "rnnoise",
        lambda e: e["aliases"].update({"rnnoise-fp32": "fp32"}),
        "missing precision",
    ),
    "bad sha256": ("rnnoise", lambda e: e["precisions"]["int8"]["model"].update(sha256="ABC"), "sha256"),
    "remote scheme": ("rnnoise", lambda e: e["precisions"]["int8"]["model"].update(uri="ftp://x/m.tflite"), "scheme"),
    "absolute path": ("rnnoise", lambda e: e["card"].update(uri="repo:///etc/passwd"), "absolute or contain"),
    "parent path": ("rnnoise", lambda e: e["card"].update(uri="repo://../x.md"), "absolute or contain"),
    "unknown precision": ("rnnoise", lambda e: e["precisions"].update(int4=e["precisions"]["int8"]), "expected one of"),
    "single with steps": ("rnnoise", lambda e: GOLDEN(e).update(steps=2), "exactly one step"),
    "resets outside a sequence": (
        "rnnoise",
        lambda e: GOLDEN(e).update(kind="batch", steps=3, resets=[1]),
        "only a sequence",
    ),
    "unknown resolver": ("rnnoise", lambda e: GOLDEN(e).update(resolver="auto"), "expected one of"),
    "explicit_state without pairs": ("rnnoise", lambda e: e["io"].update(state_pairs=[]), "explicit_state needs"),
    "pairs on a stateless model": (
        "mlperf-tiny-kws",
        lambda e: e["io"].update(state_pairs=[{"input": 0, "output": 0, "reset": "zeros", "scales_tied": False}]),
        "explicit_state needs",
    ),
    "pair out of range": ("rnnoise", lambda e: e["io"]["state_pairs"][0].update(output=9), "out of range"),
    "input paired twice": ("rnnoise", lambda e: e["io"]["state_pairs"][1].update(input=1), "paired twice"),
    "scale without zero point": (
        "rnnoise",
        lambda e: e["precisions"]["int8"]["inputs"][0].update(zero_point=None),
        "both be set",
    ),
    "short upstream revision": ("rnnoise", lambda e: e["upstream"].update(revision="fec0bb5b"), "40-hex"),
    "bad visibility": ("rnnoise", lambda e: e.update(visibility="internal"), "expected one of"),
    "output paired twice": ("rnnoise", lambda e: e["io"]["state_pairs"][1].update(output=3), "paired twice"),
    "missing key": ("rnnoise", lambda e: e.pop("tier"), "missing"),
    "missing artifact size": ("rnnoise", lambda e: e["precisions"]["int8"]["model"].pop("bytes"), "missing"),
    "reset outside the steps": (
        "rnnoise",
        lambda e: GOLDEN(e).update(kind="sequence", steps=3, resets=[3]),
        "step indices",
    ),
    "unknown dtype": (
        "rnnoise",
        lambda e: e["precisions"]["int8"]["inputs"][0].update(dtype="int4"),
        "expected one of",
    ),
    "zero-byte artifact": ("rnnoise", lambda e: e["precisions"]["int8"]["model"].update(bytes=0), ">= 1"),
    "unknown tier": ("rnnoise", lambda e: e.update(tier="trained"), "expected one of"),
    "unknown reset": ("rnnoise", lambda e: e["io"]["state_pairs"][0].update(reset="ones"), "expected one of"),
    "scales_tied not a boolean": ("rnnoise", lambda e: e["io"]["state_pairs"][0].update(scales_tied=0), "boolean"),
    "uppercase id": ("rnnoise", lambda e: e.update(id="RNNoise"), "lowercase"),
    "alias value not a string": ("rnnoise", lambda e: e["aliases"].update({"rnnoise-int8": ["int8"]}), "aliases"),
    "NaN scale": ("rnnoise", lambda e: e["precisions"]["int8"]["inputs"][0].update(scale=float("nan")), "positive"),
    "infinite scale": (
        "rnnoise",
        lambda e: e["precisions"]["int8"]["inputs"][0].update(scale=float("inf")),
        "positive",
    ),
    "boolean scale": ("rnnoise", lambda e: e["precisions"]["int8"]["inputs"][0].update(scale=True), "positive"),
    "card in lfs": ("rnnoise", lambda e: e["card"].update(uri="lfs://audio/rnnoise/README.md"), "scheme"),
    "model in repo scheme": (
        "rnnoise",
        lambda e: e["precisions"]["int8"]["model"].update(uri="repo://audio/rnnoise/model.tflite"),
        "scheme",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_parser_refuses(data, case):
    model_id, change, message = CASES[case]
    if model_id is None:
        change(data)
    else:
        change(entry(data, model_id))
    with pytest.raises(ManifestError, match=message):
        parse_manifest(data)


def test_packaged_document_parses():
    assert len(parse_manifest(PACKAGED).entries) == 9
