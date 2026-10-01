# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import http.client
import json
import os
import sys
import types

import pytest
from conftest import POINTER, entry

import helia_model_zoo as zoo
from helia_model_zoo import hydrate
from helia_model_zoo.hydrate import FetchError, fetch_file
from helia_model_zoo.manifest import FileRef, ManifestError, parse_hf, parse_manifest

COMMIT = "0" * 39 + "1"
BODY = b"model bytes"
SHA = hashlib.sha256(BODY).hexdigest()
HF = "hf://datasets/Example/models@" + "a" * 40 + "/vad/int16x8/model.tflite"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No ambient checkout, revision, overlay or install metadata; a private cache."""
    for name in ("HELIA_ZOO_ROOT", "HELIA_ZOO_REVISION", "HELIA_ZOO_OVERLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HELIA_ZOO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(hydrate, "_install", lambda: {})
    zoo.manifest.cache_clear()
    yield
    zoo.manifest.cache_clear()


class Server:
    """Records requests and serves fixed bodies in place of the network."""

    def __init__(self, monkeypatch, body=BODY):
        self.body, self.urls, self.hf = body, [], []
        monkeypatch.setattr(hydrate, "_download_url", self.url)
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = self.hf_hub_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    def url(self, url, destination, limit=None):
        self.urls.append(url)
        destination.write_bytes(self.body)

    def hf_hub_download(self, repo_id, filename, *, repo_type, revision, local_dir, token):
        self.hf.append((repo_id, filename, repo_type, revision, token))
        path = os.path.join(local_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            out.write(self.body)
        return path


def artifact(uri, body=BODY):
    return FileRef(uri, hashlib.sha256(body).hexdigest(), len(body))


def test_repository_file_downloads_at_the_revision_and_is_cached(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    ref = artifact("lfs://audio/x/model.tflite")
    path = fetch_file(ref, revision=COMMIT)
    assert path == tmp_path / "cache" / SHA / "model.tflite" and path.read_bytes() == BODY
    assert server.urls == [
        f"https://media.githubusercontent.com/media/AmbiqAI/helia-model-zoo/{COMMIT}/audio/x/model.tflite"
    ]
    assert fetch_file(ref, revision=COMMIT) == path and len(server.urls) == 1
    text = FileRef("repo://audio/x/README.md")
    assert fetch_file(text, revision=COMMIT) == tmp_path / "cache/text/repo" / COMMIT / "audio/x/README.md"
    assert server.urls[-1] == f"https://raw.githubusercontent.com/AmbiqAI/helia-model-zoo/{COMMIT}/audio/x/README.md"


def test_hydrated_checkout_is_used_without_downloading(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    (tmp_path / "repo/audio/x").mkdir(parents=True)
    (tmp_path / "repo/audio/x/model.tflite").write_bytes(BODY)
    assert fetch_file(artifact("lfs://audio/x/model.tflite"), root=tmp_path / "repo") == (
        tmp_path / "repo/audio/x/model.tflite"
    )
    monkeypatch.setenv("HELIA_ZOO_ROOT", str(tmp_path / "repo"))
    assert fetch_file(artifact("lfs://audio/x/model.tflite")).parent == tmp_path / "repo/audio/x"
    assert server.urls == []


def test_pointer_in_checkout_falls_back_to_download(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    (tmp_path / "repo/audio/x").mkdir(parents=True)
    (tmp_path / "repo/audio/x/model.tflite").write_bytes(POINTER + b"\n")
    path = fetch_file(artifact("lfs://audio/x/model.tflite"), root=tmp_path / "repo", revision=COMMIT)
    assert path.parent.name == SHA and len(server.urls) == 1


def test_changed_file_in_checkout_is_refused(monkeypatch, tmp_path):
    Server(monkeypatch)
    (tmp_path / "repo/audio/x").mkdir(parents=True)
    (tmp_path / "repo/audio/x/model.tflite").write_bytes(b"other bytes")
    with pytest.raises(FetchError, match="does not match the manifest"):
        fetch_file(artifact("lfs://audio/x/model.tflite"), root=tmp_path / "repo", revision=COMMIT)


def test_no_checkout_and_no_revision_is_explained(monkeypatch):
    Server(monkeypatch)
    with pytest.raises(FetchError, match="no hydrated checkout and no revision"):
        fetch_file(artifact("lfs://audio/x/model.tflite"))


@pytest.mark.parametrize(
    ("served", "message"), [(b"tampered!!!", "sha256 mismatch"), (b"short", "downloaded 5 bytes, expected 11")]
)
def test_wrong_download_leaves_nothing_in_the_cache(monkeypatch, tmp_path, served, message):
    Server(monkeypatch, served)
    with pytest.raises(FetchError, match=message):
        fetch_file(artifact("https://example.com/m.tflite"))
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


def test_failed_transfer_leaves_nothing_in_the_cache(monkeypatch, tmp_path):
    def broken(url, destination, limit=None):
        destination.write_bytes(b"part")
        raise OSError("connection reset")

    monkeypatch.setattr(hydrate, "_download_url", broken)
    with pytest.raises(FetchError, match="connection reset"):
        fetch_file(artifact("https://example.com/m.tflite"))
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


def test_changed_cache_entry_is_downloaded_again(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    ref = artifact("https://example.com/m.tflite")
    path = fetch_file(ref)
    path.write_bytes(b"corrupted!!")
    assert fetch_file(ref).read_bytes() == BODY and len(server.urls) == 2


def test_hf_source_uses_huggingface_hub(monkeypatch):
    server = Server(monkeypatch)
    path = fetch_file(artifact(HF))
    assert path.read_bytes() == BODY
    assert server.hf == [("Example/models", "vad/int16x8/model.tflite", "dataset", "a" * 40, None)]
    fetch_file(artifact(HF), cache=path.parent.parent / "other", anonymous=True)
    assert server.hf[-1][-1] is False


def test_hf_without_huggingface_hub_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(FetchError, match=r"install helia-model-zoo\[hf\]"):
        fetch_file(artifact(HF))


@pytest.mark.parametrize(
    "uri",
    [
        "hf://Example/models@main/model.tflite",
        "hf://Example/models@" + "a" * 39 + "/model.tflite",
        "hf://Example/models/model.tflite",
        "hf://models@" + "a" * 40 + "/model.tflite",
        "hf://datasets/Example/models@" + "a" * 40 + "/../model.tflite",
        "hf://datasets/Example/models@" + "A" * 40 + "/model.tflite",
    ],
)
def test_unpinned_or_malformed_hf_uri_is_refused(data, uri):
    entry(data, "rnnoise")["precisions"]["int8"]["model"]["uri"] = uri
    with pytest.raises(ManifestError):
        parse_manifest(data)


def test_hf_uri_forms():
    assert parse_hf("hf://Org/repo@" + "b" * 40 + "/a/b.bin").repo_type == "model"
    assert parse_hf("hf://spaces/Org/repo@" + "b" * 40 + "/a").repo_type == "space"


def test_https_text_needs_a_sha256(data):
    entry(data, "rnnoise")["card"] = {"uri": "https://example.com/card.md"}
    with pytest.raises(ManifestError, match="needs a sha256"):
        parse_manifest(data)
    entry(data, "rnnoise")["card"]["sha256"] = SHA
    assert parse_manifest(data).get("rnnoise").card_file.sha256 == SHA


def test_installed_revision_and_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(hydrate, "_install", lambda: {"url": "file:///x", "vcs_info": {"commit_id": COMMIT}})
    assert zoo.revision() == COMMIT and hydrate.local_checkout() is None
    monkeypatch.setenv("HELIA_ZOO_REVISION", "f" * 40)
    assert zoo.revision() == "f" * 40
    monkeypatch.setattr(hydrate, "_install", lambda: {"url": f"file://{tmp_path}", "dir_info": {"editable": True}})
    assert hydrate.local_checkout() == tmp_path


def _overlay(tmp_path, data, model_id="private-vad", alias="private-vad-int16x8"):
    item = json.loads(json.dumps(entry(data, "rnnoise")))
    item.update(id=model_id, aliases={alias: "int8"}, visibility="private")
    item["precisions"]["int8"]["model"].update(uri=HF, sha256=SHA, bytes=len(BODY))
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": "helia-model-zoo/manifest@2", "entries": [item]}))
    return path


def test_overlay_joins_the_packaged_manifest(monkeypatch, tmp_path, data):
    server = Server(monkeypatch)
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(_overlay(tmp_path, data)))
    assert {"rnnoise", "private-vad"} <= {e.id for e in zoo.entries()}
    assert zoo.fetch("private-vad-int16x8").read_bytes() == BODY
    assert zoo.resolve("zoo://private-vad/int8").read_bytes() == BODY
    assert server.hf[0][0] == "Example/models"


def test_overlay_may_not_reuse_a_packaged_name(monkeypatch, tmp_path, data):
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(_overlay(tmp_path, data, alias="rnnoise-int8")))
    with pytest.raises(ManifestError, match="already used"):
        zoo.manifest()


def test_overlay_location_forms(monkeypatch, tmp_path, data):
    server = Server(monkeypatch, _overlay(tmp_path, data).read_bytes())
    overlay = zoo.load_overlay("hf://datasets/Example/index@" + "c" * 40 + "/manifest.json")
    assert overlay.entries[0].id == "private-vad" and server.hf[0][3] == "c" * 40
    with pytest.raises(ManifestError, match="40-hex"):
        zoo.load_overlay("hf://datasets/Example/index@main/manifest.json")
    with pytest.raises(ManifestError, match="expected a local path or an hf:// URI"):
        zoo.load_overlay("https://example.com/manifest.json")


def test_resolve_and_precision_choice(monkeypatch, tmp_path):
    Server(monkeypatch)
    with pytest.raises(ValueError, match="zoo://"):
        zoo.resolve("rnnoise")
    with pytest.raises(KeyError, match="no precision 'fp32'"):
        zoo.resolve("zoo://rnnoise/fp32")


def test_entry_api_from_a_checkout(root):
    item = zoo.get("rnnoise")
    assert item.fetch(root=root) == root / "audio/rnnoise/model.tflite"
    golden = item.golden(root=root)
    assert len(golden.inputs) == 4 and len(golden.outputs) == 5 and golden.meta.resolver == "builtin_ref"
    assert item.card(root=root) == root / "audio/rnnoise/README.md"
    assert zoo.fetch("rnnoise-int8", root=root) == root / "audio/rnnoise/model.tflite"


@pytest.mark.skipif(os.environ.get("HELIA_ZOO_LIVE") != "1", reason="set HELIA_ZOO_LIVE=1 to reach Hugging Face")
def test_live_public_hugging_face_file():
    pytest.importorskip("huggingface_hub")
    ref = FileRef(
        "hf://hf-internal-testing/tiny-random-bert@f171d7baecaf37b5da5a3616d8833b9969753535/config.json",
        "54f9f0001ec62b6a53072a7c9d1b630e346ae51fca14a5799e54f0d918281872",
        548,
    )
    assert fetch_file(ref, anonymous=True).stat().st_size == 548


def test_public_remote_artifact_must_download_without_credentials(monkeypatch, root, data):
    from helia_model_zoo.validate import ValidationError, validate

    model = entry(data, "mlperf-tiny-kws")["precisions"]["int8"]["model"]
    model["uri"] = "https://example.com/kws.tflite"
    body = (root / "audio/mlperf-tiny/kws_ref/model.tflite").read_bytes()
    calls = []

    def served(url, destination, limit=None):
        calls.append(url)
        destination.write_bytes(body)

    monkeypatch.setattr(hydrate, "_download_url", served)
    only = {"schema": data["schema"], "entries": [entry(data, "mlperf-tiny-kws")]}
    validate(root, parse_manifest(only), replay=True)
    assert calls == ["https://example.com/kws.tflite"]

    def refused(url, destination, limit=None):
        raise OSError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(hydrate, "_download_url", refused)
    model["sha256"] = "f" * 64
    with pytest.raises(ValidationError) as caught:
        validate(root, parse_manifest(only))
    assert caught.value.problems == [
        "entry mlperf-tiny-kws.precisions.int8.model: could not download https://example.com/kws.tflite: "
        "OSError: HTTP Error 401: Unauthorized (fetched without credentials, as a public entry must be)"
    ]


def test_private_validation_may_use_credentials(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    server = Server(monkeypatch)
    overlay = parse_manifest(json.loads(_overlay(tmp_path, data).read_text()))
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, overlay, signatures=False)
    assert caught.value.problems[0].startswith("entry private-vad: visibility is 'private'")
    with pytest.raises(ValidationError):
        validate(tmp_path, overlay, signatures=False, public=False)
    assert server.hf and server.hf[-1][-1] is None


def test_cli_fetch(monkeypatch, root, tmp_path, capsys):
    from helia_model_zoo.cli import main

    assert main(["fetch", "rnnoise-int8", "--golden", "--card", "--root", str(root)]) == 0
    assert capsys.readouterr().out.split() == [
        str(root / "audio/rnnoise/model.tflite"),
        str(root / "audio/rnnoise/golden.npz"),
        str(root / "audio/rnnoise/README.md"),
    ]
    assert main(["fetch", "rnnoise", "--precision", "fp32", "--root", str(root)]) == 1
    assert "no precision 'fp32'" in capsys.readouterr().err
    assert main(["fetch", "rnnoise"]) == 1
    assert "no hydrated checkout and no revision" in capsys.readouterr().err


def test_public_hf_artifact_is_fetched_without_a_token(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    server = Server(monkeypatch)
    item = entry(data, "rnnoise")
    item["precisions"]["int8"]["model"].update(uri=HF, sha256=SHA, bytes=len(BODY))
    with pytest.raises(ValidationError):
        validate(tmp_path, parse_manifest({"schema": data["schema"], "entries": [item]}), signatures=False)
    assert server.hf == [("Example/models", "vad/int16x8/model.tflite", "dataset", "a" * 40, False)]


def test_alias_names_its_precision_among_several(monkeypatch, tmp_path, data):
    Server(monkeypatch)
    path = _overlay(tmp_path, data, alias="private-vad-wide")
    overlay = json.loads(path.read_text())
    precisions = overlay["entries"][0]["precisions"]
    precisions["int16x8"] = json.loads(json.dumps(precisions["int8"]))
    precisions["int16x8"]["model"]["uri"] = HF.replace("model.tflite", "wide.tflite")
    overlay["entries"][0]["aliases"] = {"private-vad-wide": "int16x8"}
    path.write_text(json.dumps(overlay))
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(path))
    assert zoo.fetch("private-vad-wide").name == "wide.tflite"
    assert zoo.fetch("private-vad-wide", "int8").name == "model.tflite"
    with pytest.raises(KeyError, match="name one"):
        zoo.fetch("private-vad")


def test_target_appears_only_after_verification(monkeypatch, tmp_path):
    ref = artifact("https://example.com/m.tflite")
    target = tmp_path / "cache" / SHA / "m.tflite"

    def checking(url, destination, limit=None):
        assert destination != target and destination.parent == target.parent and not target.exists()
        destination.write_bytes(BODY)

    monkeypatch.setattr(hydrate, "_download_url", checking)
    assert fetch_file(ref) == target and [p.name for p in target.parent.iterdir()] == ["m.tflite"]


def test_hf_download_leaves_only_the_verified_file(monkeypatch, tmp_path):
    Server(monkeypatch)
    fetch_file(artifact(HF))
    assert sorted(str(p.relative_to(tmp_path / "cache")) for p in (tmp_path / "cache").rglob("*")) == [
        SHA,
        f"{SHA}/model.tflite",
    ]


def test_explicit_revision_wins_over_the_installed_one(monkeypatch):
    server = Server(monkeypatch)
    monkeypatch.setattr(hydrate, "_install", lambda: {"vcs_info": {"commit_id": "f" * 40}})
    fetch_file(artifact("lfs://audio/x/model.tflite"), revision=COMMIT)
    assert COMMIT in server.urls[0] and "f" * 40 not in server.urls[0]
    server.body = b"other model"
    fetch_file(artifact("lfs://audio/y/model.tflite", b"other model"))
    assert "f" * 40 in server.urls[1]


@pytest.mark.parametrize("revision", ["main", "../../outside", "A" * 40, "0" * 39])
def test_revision_must_be_a_full_commit(monkeypatch, revision):
    server = Server(monkeypatch)
    with pytest.raises(FetchError, match="full 40-hex commit"):
        fetch_file(FileRef("repo://audio/x/README.md"), revision=revision)
    assert server.urls == []


def test_repository_paths_are_quoted(monkeypatch):
    server = Server(monkeypatch)
    fetch_file(artifact("lfs://audio/my model/model.tflite"), revision=COMMIT)
    assert server.urls[0].endswith(f"/{COMMIT}/audio/my%20model/model.tflite")


def test_download_stops_past_the_expected_size(monkeypatch, tmp_path):
    class Endless:
        def read(self, n):
            return b"x" * n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(hydrate.urllib.request, "urlopen", lambda url, timeout: Endless())
    with pytest.raises(FetchError, match="more than the expected 11 bytes"):
        fetch_file(artifact("https://example.com/m.tflite"))
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"x"), ValueError("bad url")])
def test_transport_errors_become_fetch_errors(monkeypatch, error):
    def failing(url, timeout):
        raise error

    monkeypatch.setattr(hydrate.urllib.request, "urlopen", failing)
    with pytest.raises(FetchError, match="could not download https://example.com/m.tflite"):
        fetch_file(artifact("https://example.com/m.tflite"))


def test_public_check_ignores_a_warm_cache(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    item = entry(data, "rnnoise")
    item["card"] = {"uri": "https://example.com/card.md", "sha256": SHA}
    only = parse_manifest({"schema": data["schema"], "entries": [item]})
    Server(monkeypatch)
    fetch_file(only.entries[0].card_file)  # now cached in HELIA_ZOO_CACHE

    def refused(url, destination, limit=None):
        raise OSError("HTTP Error 401: Unauthorized")

    monkeypatch.setattr(hydrate, "_download_url", refused)
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, only, signatures=False)
    assert any("card: could not download https://example.com/card.md" in p for p in caught.value.problems)


def test_https_parent_segment_is_refused(data):
    entry(data, "rnnoise")["precisions"]["int8"]["model"]["uri"] = "https://example.com/a/../m.tflite"
    with pytest.raises(ManifestError, match="without '..'"):
        parse_manifest(data)


def test_missing_overlay_file_is_a_manifest_error(tmp_path):
    with pytest.raises(ManifestError, match="no such file"):
        zoo.load_overlay(tmp_path / "missing.json")


def test_cli_fetch_uses_the_alias_precision(monkeypatch, tmp_path, data, capsys):
    from helia_model_zoo.cli import main

    test_alias_names_its_precision_among_several(monkeypatch, tmp_path, data)
    capsys.readouterr()
    assert main(["fetch", "private-vad-wide"]) == 0
    assert capsys.readouterr().out.strip().endswith("wide.tflite")
