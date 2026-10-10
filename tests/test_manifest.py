# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import json
from importlib import resources

import pytest
from conftest import PACKAGED, REPO, entry

import helia_model_zoo as zoo
from helia_model_zoo.manifest import ManifestError, load_manifest, packaged_models, parse_records


def test_records_hold_the_v1_corpus():
    v1 = json.loads((REPO / "corpus-manifest-v1.json").read_text())
    models = {p.model.path: p for r in zoo.records() for p in r.precisions.values()}
    assert (len(zoo.records()), len(v1["entries"])) == (11, 9)
    for item in v1["entries"]:
        assert models[item["model"]].model.sha256 == item["model_sha256"]
    assert all(r.visibility == "public" for r in zoo.records())


def test_get_by_id():
    assert zoo.get("rnnoise").precision().name == "a8w8"
    assert zoo.get("dfnet2").precision().name == "a16w8"
    with pytest.raises(KeyError):
        zoo.get("no-such-model")


def test_state_pairs_name_tensors():
    rnnoise = zoo.get("rnnoise")
    assert rnnoise.io.streaming == "explicit_state"
    assert rnnoise.precision().pair_indices(rnnoise.io) == ((1, 3), (2, 2), (3, 0))
    assert [(p.input, p.output) for p in rnnoise.io.state_pairs][0] == ("vad_gru_prev_state_int8", "Identity_3_int8")
    assert zoo.get("dfnet2").precision().pair_indices(zoo.get("dfnet2").io) == ((2, 4),)
    assert zoo.get("mlperf-tiny-kws").io.state_pairs == ()


def test_records_resolve_inside_their_model_directory():
    record = zoo.get("rnnoise")
    assert record.card_file.uri == "repo://models/rnnoise/README.md"
    assert record.precision().model.uri == "lfs://models/rnnoise/a8w8/model.tflite"
    assert record.precision().golden.file.uri == "lfs://models/rnnoise/a8w8/golden.npz"
    assert (record.precision().golden.reference_runtime, record.precision().golden.reference_runtime_version) == (
        "ai-edge-litert",
        "2.1.2",
    )


def test_a_checkout_reads_its_own_models_directory():
    assert packaged_models() == REPO / "models"
    assert {r.id for r in load_manifest(REPO / "models").records} == set(PACKAGED)


def test_a_malformed_record_names_its_file(tmp_path):
    (tmp_path / "models/broken").mkdir(parents=True)
    for text in ("{", "[" * 100_000):
        (tmp_path / "models/broken/record.json").write_text(text)
        with pytest.raises(ManifestError, match="broken/record.json"):
            load_manifest(tmp_path / "models")


def test_an_installed_package_reads_its_shipped_records(monkeypatch, tmp_path):
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(resources, "files", lambda package: tmp_path)
    assert packaged_models() == tmp_path / "models"


PRECISION = lambda e: e["precisions"]["a8w8"]  # noqa: E731
GOLDEN = lambda e: PRECISION(e)["golden"]  # noqa: E731
HF = "hf://Example/models@" + "a" * 40 + "/m.tflite"
CASES = {
    "schema": (lambda e: e.update(schema="helia-model-zoo/manifest@2"), "unsupported schema"),
    "unknown key": (lambda e: e.update(extra=1), "unknown keys"),
    "removed alias field": (lambda e: e.update(aliases={}), "unknown keys"),
    "removed tier field": (lambda e: e.update(tier="converted"), "unknown keys"),
    "id differs from directory": (lambda e: e.update(id="wav2letter"), "directory name"),
    "uppercase id": (lambda e: e.update(id="RNNoise"), "directory name"),
    "bad sha256": (lambda e: PRECISION(e)["model"].update(sha256="ABC"), "sha256"),
    "path and uri": (lambda e: PRECISION(e)["model"].update(uri=HF), "exactly one of path or uri"),
    "neither path nor uri": (lambda e: PRECISION(e)["model"].pop("path"), "exactly one of path or uri"),
    "http uri": (
        lambda e: (PRECISION(e)["model"].pop("path"), PRECISION(e)["model"].update(uri="http://x/m.tflite")),
        "hf://",
    ),
    "unpinned hf uri": (
        lambda e: (PRECISION(e)["model"].pop("path"), PRECISION(e)["model"].update(uri="hf://Example/m@main/m.tflite")),
        "40-hex",
    ),
    "absolute path": (lambda e: e.update(card="/etc/passwd"), "inside the model directory"),
    "dot-led path": (lambda e: PRECISION(e)["model"].update(path="./a8w8/model.tflite"), "inside the model directory"),
    "empty path part": (
        lambda e: PRECISION(e)["model"].update(path="a8w8//model.tflite"),
        "inside the model directory",
    ),
    "NUL in a path": (
        lambda e: PRECISION(e)["model"].update(path="a8w8/model\x00.tflite"),
        "inside the model directory",
    ),
    "hf path with a leading slash": (
        lambda e: (
            PRECISION(e)["model"].pop("path"),
            PRECISION(e)["model"].update(uri=HF.replace("/m.tflite", "//m.tflite")),
        ),
        "expected hf://",
    ),
    "parent path": (lambda e: e.update(card="../wav2letter/README.md"), "inside the model directory"),
    "scheme in a path": (lambda e: PRECISION(e)["model"].update(path="lfs://x.tflite"), "inside the model directory"),
    "old precision name": (lambda e: e["precisions"].update(int8=e["precisions"].pop("a8w8")), "expected one of"),
    "single with steps": (lambda e: GOLDEN(e).update(steps=2), "exactly one step"),
    "resets outside a sequence": (lambda e: GOLDEN(e).update(kind="batch", steps=3, resets=[1]), "only a sequence"),
    "reset outside the steps": (lambda e: GOLDEN(e).update(kind="sequence", steps=3, resets=[3]), "step indices"),
    "unknown resolver": (lambda e: GOLDEN(e).update(resolver="auto"), "expected one of"),
    "runtime without a version": (lambda e: GOLDEN(e).update(runtime="ai-edge-litert"), "<distribution>==<version>"),
    "golden missing a field": (lambda e: GOLDEN(e).pop("resolver"), "missing"),
    "explicit_state without pairs": (lambda e: e["io"].update(state_pairs=[]), "explicit_state needs"),
    "pair names a missing tensor": (lambda e: e["io"]["state_pairs"][0].update(out="nope"), "no tensor named 'nope'"),
    "input paired twice": (
        lambda e: e["io"]["state_pairs"][1].update({"in": "vad_gru_prev_state_int8"}),
        "paired twice",
    ),
    "output paired twice": (lambda e: e["io"]["state_pairs"][1].update(out="Identity_3_int8"), "paired twice"),
    "pair by index": (lambda e: e["io"]["state_pairs"][0].update(input=1), "unknown keys"),
    "duplicate tensor name": (
        lambda e: PRECISION(e)["inputs"].append(dict(PRECISION(e)["inputs"][0])),
        "names must be unique",
    ),
    "duplicate output name": (
        lambda e: PRECISION(e)["outputs"].append(dict(PRECISION(e)["outputs"][0])),
        "names must be unique",
    ),
    "scale without zero point": (lambda e: PRECISION(e)["inputs"][0].update(zero_point=None), "both be set"),
    "NaN scale": (lambda e: PRECISION(e)["inputs"][0].update(scale=float("nan")), "positive"),
    "infinite scale": (lambda e: PRECISION(e)["inputs"][0].update(scale=float("inf")), "positive"),
    "boolean scale": (lambda e: PRECISION(e)["inputs"][0].update(scale=True), "positive"),
    "huge integer scale": (lambda e: PRECISION(e)["inputs"][0].update(scale=10**400), "positive"),
    "unknown dtype": (lambda e: PRECISION(e)["inputs"][0].update(dtype="int4"), "expected one of"),
    "short upstream revision": (lambda e: e["upstream"].update(revision="fec0bb5b"), "40-hex"),
    "bad visibility": (lambda e: e.update(visibility="internal"), "expected one of"),
    "missing key": (lambda e: e.pop("task"), "missing"),
    "missing artifact size": (lambda e: PRECISION(e)["model"].pop("bytes"), "missing"),
    "zero-byte artifact": (lambda e: PRECISION(e)["model"].update(bytes=0), ">= 1"),
    "licence with a removed field": (lambda e: e["license"].update(reference="README.md"), "unknown keys"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_parser_refuses(data, case):
    change, message = CASES[case]
    change(entry(data, "rnnoise"))
    with pytest.raises(ManifestError, match=message):
        parse_records(data)


def test_a8w4_is_a_precision(data):
    record = entry(data, "rnnoise")
    record["precisions"]["a8w4"] = record["precisions"].pop("a8w8")
    assert parse_records(data).get("rnnoise").precision().name == "a8w4"


def test_hf_artifacts_and_cards_are_accepted(data):
    record = entry(data, "rnnoise")
    record["card"] = "hf://datasets/Example/cards@" + "b" * 40 + "/rnnoise/README.md"
    model = record["precisions"]["a8w8"]["model"]
    model.pop("path")
    model["uri"] = HF
    parsed = parse_records(data).get("rnnoise")
    assert parsed.precision().model.uri == HF and parsed.card_file.uri.startswith("hf://datasets/Example/cards@")


def test_repository_records_parse():
    assert len(parse_records(PACKAGED).records) == 11
