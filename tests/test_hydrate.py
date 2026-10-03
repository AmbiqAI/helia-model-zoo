# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import http.client
import json
import os
import sys
import types
from pathlib import Path

import pytest
from conftest import POINTER, entry

import helia_model_zoo as zoo
from helia_model_zoo import hydrate
from helia_model_zoo.hydrate import FetchError, fetch_file
from helia_model_zoo.manifest import FileRef, ManifestError, parse_hf, parse_records

COMMIT = "0" * 39 + "1"
BODY = b"model bytes"
SHA = hashlib.sha256(BODY).hexdigest()
HF = "hf://datasets/Example/models@" + "a" * 40 + "/vad/a16w8/model.tflite"
LFS = "lfs://models/x/a8w8/model.tflite"


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
        server = self

        class HfApi:
            def list_repo_files(self, repo_id, *, repo_type, revision, token):
                server.listed.append((repo_id, repo_type, revision, token))
                return server.files

        module.HfApi = HfApi
        self.listed, self.files = [], []
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
    with pytest.raises(FetchError, match="does not match the record"):
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
        fetch_file(artifact(LFS), revision=COMMIT)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


def test_failed_transfer_leaves_nothing_in_the_cache(monkeypatch, tmp_path):
    def broken(url, destination, limit=None):
        destination.write_bytes(b"part")
        raise OSError("connection reset")

    monkeypatch.setattr(hydrate, "_download_url", broken)
    with pytest.raises(FetchError, match="connection reset"):
        fetch_file(artifact(LFS), revision=COMMIT)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


def test_changed_cache_entry_is_downloaded_again(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    ref = artifact(LFS)
    path = fetch_file(ref, revision=COMMIT)
    path.write_bytes(b"corrupted!!")
    assert fetch_file(ref, revision=COMMIT).read_bytes() == BODY and len(server.urls) == 2


def test_hf_source_uses_huggingface_hub(monkeypatch):
    server = Server(monkeypatch)
    path = fetch_file(artifact(HF))
    assert path.read_bytes() == BODY
    assert server.hf == [("Example/models", "vad/a16w8/model.tflite", "dataset", "a" * 40, None)]
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
    entry(data, "rnnoise")["precisions"]["a8w8"]["model"]["uri"] = uri
    with pytest.raises(ManifestError):
        parse_records(data)


def test_hf_uri_forms():
    assert parse_hf("hf://Org/repo@" + "b" * 40 + "/a/b.bin").repo_type == "model"
    assert parse_hf("hf://spaces/Org/repo@" + "b" * 40 + "/a").repo_type == "space"


def test_other_schemes_are_not_fetched(monkeypatch):
    server = Server(monkeypatch)
    with pytest.raises(FetchError, match="unsupported source"):
        fetch_file(artifact("https://example.com/m.tflite"), revision=COMMIT)
    assert server.urls == []


def test_installed_revision_and_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(hydrate, "_install", lambda: {"url": "file:///x", "vcs_info": {"commit_id": COMMIT}})
    assert zoo.revision() == COMMIT and hydrate.local_checkout() is None
    monkeypatch.setenv("HELIA_ZOO_REVISION", "f" * 40)
    assert zoo.revision() == "f" * 40
    monkeypatch.setattr(hydrate, "_install", lambda: {"url": f"file://{tmp_path}", "dir_info": {"editable": True}})
    assert hydrate.local_checkout() == tmp_path


def _overlay(tmp_path, data, model_id="private-vad"):
    """An overlay directory holding one private record whose files live on Hugging Face."""
    item = json.loads(json.dumps(entry(data, "rnnoise")))
    item.update(id=model_id, visibility="private", card=HF.replace("a16w8/model.tflite", "README.md"))
    precision = item["precisions"]["a8w8"]
    for key in ("model", "golden"):
        precision[key]["uri"] = HF.replace("model.tflite", Path(precision[key].pop("path")).name)
    precision["model"].update(sha256=SHA, bytes=len(BODY))
    record = tmp_path / "overlay/models" / model_id / "record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps(item))
    return tmp_path / "overlay"


def test_overlay_joins_the_packaged_records(monkeypatch, tmp_path, data):
    server = Server(monkeypatch)
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(_overlay(tmp_path, data)))
    assert {"rnnoise", "private-vad"} <= {r.id for r in zoo.records()}
    assert zoo.fetch("private-vad").read_bytes() == BODY
    assert zoo.resolve("zoo://private-vad/a8w8").read_bytes() == BODY
    assert server.hf[0][0] == "Example/models"


def test_overlay_may_not_reuse_a_packaged_id(monkeypatch, tmp_path, data):
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(_overlay(tmp_path, data, model_id="rnnoise")))
    with pytest.raises(ManifestError, match="already used"):
        zoo.manifest()


def test_overlay_records_give_files_as_hf_uris(tmp_path, data):
    overlay = _overlay(tmp_path, data)
    record = overlay / "models/private-vad/record.json"
    item = json.loads(record.read_text())
    item["card"] = "README.md"
    record.write_text(json.dumps(item))
    with pytest.raises(ManifestError, match="private-vad: give every file as an hf:// URI"):
        zoo.load_overlay(overlay)


def test_overlay_location_forms(monkeypatch, tmp_path, data):
    record = (_overlay(tmp_path, data) / "models/private-vad/record.json").read_bytes()
    server = Server(monkeypatch, record)
    server.files = ["README.md", "record.json", "models/private-vad/record.json", "models/private-vad/a8w8/record.json"]
    overlay = zoo.load_overlay("hf://datasets/Example/index@" + "c" * 40)
    assert [r.id for r in overlay.records] == ["private-vad"]
    assert server.listed == [("Example/index", "dataset", "c" * 40, None)]
    assert server.hf == [("Example/index", "models/private-vad/record.json", "dataset", "c" * 40, None)]
    zoo.load_overlay("hf://datasets/Example/index@" + "c" * 40, anonymous=True)
    assert server.listed[-1] == ("Example/index", "dataset", "c" * 40, False)
    for location in (
        "hf://datasets/Example/index@main",
        "hf://datasets/Example/index@" + "c" * 40 + "/sub",
        "hf://Example/index@" + "c" * 40,
    ):
        with pytest.raises(ManifestError, match="as an overlay root"):
            zoo.load_overlay(location)
    with pytest.raises(ManifestError, match="expected a local directory or an hf:// dataset root"):
        zoo.load_overlay("https://example.com/overlay")
    server.files = ["models/.cache/record.json"]
    with pytest.raises(ManifestError, match="overlay: expected hf://"):
        zoo.load_overlay("hf://datasets/Example/index@" + "c" * 40)
    server.files = ["README.md"]
    with pytest.raises(ManifestError, match="no models/<id>/record.json"):
        zoo.load_overlay("hf://datasets/Example/index@" + "c" * 40)


def test_resolve_and_precision_choice(monkeypatch, tmp_path):
    Server(monkeypatch)
    with pytest.raises(ValueError, match="zoo://"):
        zoo.resolve("rnnoise")
    with pytest.raises(KeyError, match="no precision 'fp32'"):
        zoo.resolve("zoo://rnnoise/fp32")


def test_entry_api_from_a_checkout(root):
    item = zoo.get("rnnoise")
    assert item.fetch(root=root) == root / "models/rnnoise/a8w8/model.tflite"
    golden = item.golden(root=root)
    assert len(golden.inputs) == 4 and len(golden.outputs) == 5 and golden.meta.resolver == "builtin_ref"
    assert item.card(root=root) == root / "models/rnnoise/README.md"
    assert zoo.fetch("rnnoise", root=root) == root / "models/rnnoise/a8w8/model.tflite"


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

    body = (root / "models/mlperf-tiny-kws/a8w8/model.tflite").read_bytes()
    server = Server(monkeypatch, body)
    model = entry(data, "mlperf-tiny-kws")["precisions"]["a8w8"]["model"]
    model.pop("path")
    model["uri"] = HF
    only = {"mlperf-tiny-kws": entry(data, "mlperf-tiny-kws")}
    validate(root, parse_records(only), replay=True)
    assert server.hf == [("Example/models", "vad/a16w8/model.tflite", "dataset", "a" * 40, False)]

    def refused(*args, **kwargs):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", refused)
    model["sha256"] = "f" * 64
    with pytest.raises(ValidationError) as caught:
        validate(root, parse_records(only))
    assert caught.value.problems == [
        f"record mlperf-tiny-kws.precisions.a8w8.model: could not download {HF}: OSError "
        "(fetched without credentials, as a public record must be)"
    ]


def test_private_validation_may_use_credentials(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    server = Server(monkeypatch)
    overlay = zoo.load_overlay(_overlay(tmp_path, data))
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, overlay, signatures=False)
    assert caught.value.problems[0].startswith("record private-vad: visibility is 'private'")
    with pytest.raises(ValidationError):
        validate(tmp_path, overlay, signatures=False, public=False)
    assert server.hf and server.hf[-1][-1] is None


def test_cli_fetch(monkeypatch, root, tmp_path, capsys):
    from helia_model_zoo.cli import main

    assert main(["fetch", "rnnoise", "--golden", "--card", "--root", str(root)]) == 0
    assert capsys.readouterr().out.split() == [
        str(root / "models/rnnoise/a8w8/model.tflite"),
        str(root / "models/rnnoise/a8w8/golden.npz"),
        str(root / "models/rnnoise/README.md"),
    ]
    assert main(["fetch", "rnnoise", "--precision", "fp32", "--root", str(root)]) == 1
    assert "no precision 'fp32'" in capsys.readouterr().err
    assert main(["fetch", "rnnoise"]) == 1
    assert "no hydrated checkout and no revision" in capsys.readouterr().err


def test_public_hf_artifact_is_fetched_without_a_token(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    server = Server(monkeypatch)
    model = entry(data, "rnnoise")["precisions"]["a8w8"]["model"]
    model.pop("path")
    model.update(uri=HF, sha256=SHA, bytes=len(BODY))
    with pytest.raises(ValidationError):
        validate(tmp_path, parse_records({"rnnoise": entry(data, "rnnoise")}), signatures=False)
    assert server.hf == [("Example/models", "vad/a16w8/model.tflite", "dataset", "a" * 40, False)]


def test_unnamed_precision_needs_one_among_several(monkeypatch, tmp_path, data):
    Server(monkeypatch)
    root = _overlay(tmp_path, data)
    path = root / "models/private-vad/record.json"
    record = json.loads(path.read_text())
    record["precisions"]["a16w8"] = json.loads(json.dumps(record["precisions"]["a8w8"]))
    record["precisions"]["a16w8"]["model"]["uri"] = HF.replace("model.tflite", "wide.tflite")
    path.write_text(json.dumps(record))
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(root))
    assert zoo.fetch("private-vad", "a16w8").name == "wide.tflite"
    assert zoo.fetch("private-vad", "a8w8").name == "model.tflite"
    with pytest.raises(KeyError, match="name one"):
        zoo.fetch("private-vad")


def test_target_appears_only_after_verification(monkeypatch, tmp_path):
    ref = artifact(LFS)
    target = tmp_path / "cache" / SHA / "model.tflite"

    def checking(url, destination, limit=None):
        assert destination != target and destination.parent == target.parent and not target.exists()
        destination.write_bytes(BODY)

    monkeypatch.setattr(hydrate, "_download_url", checking)
    assert fetch_file(ref, revision=COMMIT) == target and [p.name for p in target.parent.iterdir()] == ["model.tflite"]


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
        fetch_file(artifact(LFS), revision=COMMIT)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"x"), ValueError("bad url")])
def test_transport_errors_become_fetch_errors(monkeypatch, error):
    def failing(url, timeout):
        raise error

    monkeypatch.setattr(hydrate.urllib.request, "urlopen", failing)
    with pytest.raises(FetchError, match="could not download https://media.githubusercontent.com/"):
        fetch_file(artifact(LFS), revision=COMMIT)


def test_public_check_ignores_a_warm_cache(monkeypatch, tmp_path, data):
    from helia_model_zoo.validate import ValidationError, validate

    Server(monkeypatch)
    entry(data, "rnnoise")["card"] = "hf://datasets/Example/cards@" + "b" * 40 + "/rnnoise.md"
    only = parse_records({"rnnoise": entry(data, "rnnoise")})
    fetch_file(only.get("rnnoise").card_file)  # now cached in HELIA_ZOO_CACHE

    def refused(*args, **kwargs):
        raise OSError("401 Unauthorized")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "hf_hub_download", refused)
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, only, signatures=False)
    assert any("card: could not download hf://datasets/Example/cards@" in p for p in caught.value.problems)


def test_overlay_without_records_is_a_manifest_error(tmp_path):
    with pytest.raises(ManifestError, match="no models/ directory"):
        zoo.load_overlay(tmp_path / "missing")


def test_cli_fetch_names_the_precision(monkeypatch, tmp_path, data, capsys):
    from helia_model_zoo.cli import main

    test_unnamed_precision_needs_one_among_several(monkeypatch, tmp_path, data)
    capsys.readouterr()
    assert main(["fetch", "private-vad", "--precision", "a16w8"]) == 0
    assert capsys.readouterr().out.strip().endswith("wide.tflite")
