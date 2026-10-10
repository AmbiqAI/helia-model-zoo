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
from helia_model_zoo.cli import inventory_row
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
        fetch_file(artifact("http://example.com/m.tflite"), revision=COMMIT)
    assert server.urls == []


HTTPS = "https://example.com/pinned/model.tflite"


def test_https_record_fetches_and_revalidates_cached_artifact(data, monkeypatch, tmp_path):
    server = Server(monkeypatch)
    model = entry(data, "rnnoise")["precisions"]["a8w8"]["model"]
    model.pop("path")
    model.update(uri=HTTPS, sha256=SHA, bytes=len(BODY))
    record = parse_records(data).get("rnnoise")
    assert inventory_row(record)["hosting"] == ["git-lfs", "https:example.com"]
    path = record.fetch("a8w8", cache=tmp_path, anonymous=True)
    assert path == tmp_path / SHA / "model.tflite" and path.read_bytes() == BODY
    assert server.urls == [HTTPS]
    assert record.fetch("a8w8", cache=tmp_path) == path and server.urls == [HTTPS]
    path.write_bytes(b"corrupted!!")
    assert record.fetch("a8w8", cache=tmp_path).read_bytes() == BODY
    assert server.urls == [HTTPS, HTTPS]


@pytest.mark.parametrize(
    "uri",
    [
        "https:///model.tflite",
        "https://example.com/",
        "https://user:secret@example.com/m.tflite",
        "https://example.com/m.tflite?token=secret",
        "https://example.com/m.tflite#part",
        "https://example.com/m.tflite?",
        "https://example.com/m.tflite#",
        "https://example.com/%2e%2e/m.tflite",
        "https://example.com/%2E/m.tflite",
        "https://example.com/p%2f../m.tflite",
        "https://example.com/../m.tflite",
        "https://example.com:bad/m.tflite",
        "https://example.com/\nm.tflite",
        "https://example.com/p\\m.tflite",
    ],
)
def test_invalid_https_artifact_is_refused_before_cache_or_network(data, monkeypatch, tmp_path, uri):
    server = Server(monkeypatch)
    model = entry(data, "rnnoise")["precisions"]["a8w8"]["model"]
    model.pop("path")
    model.update(uri=uri)
    with pytest.raises(ManifestError, match="HTTPS file URL"):
        parse_records(data)
    with pytest.raises(FetchError, match="HTTPS file URL"):
        fetch_file(artifact(uri), cache=tmp_path)
    assert server.urls == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("sha", "size"),
    [(None, len(BODY)), ("not-a-sha", len(BODY)), (SHA, None), (SHA, 0), (SHA, True)],
)
def test_https_requires_identity_even_when_constructed_directly(monkeypatch, tmp_path, sha, size):
    server = Server(monkeypatch)
    with pytest.raises(FetchError, match="require a valid SHA-256"):
        fetch_file(FileRef(HTTPS, sha, size), cache=tmp_path)
    assert server.urls == [] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("body", [b"tampered!!!", b"short"])
def test_https_mismatching_bytes_never_enter_cache(monkeypatch, tmp_path, body):
    Server(monkeypatch, body)
    with pytest.raises(FetchError):
        fetch_file(artifact(HTTPS), cache=tmp_path)
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


@pytest.mark.parametrize("url", ["http://example.com/model.tflite", "https://user:secret@example.com/model.tflite"])
def test_https_redirect_refuses_downgrade_or_credentials(url):
    request = hydrate.urllib.request.Request(HTTPS)
    with pytest.raises(FetchError, match="refused a non-HTTPS or credential-bearing redirect"):
        hydrate._HTTPSRedirect().redirect_request(request, None, 302, "Found", {}, url)


def test_https_redirect_allows_anonymous_https_cdn():
    request = hydrate.urllib.request.Request(HTTPS)
    target = "https://cdn.example.com/model.tflite?signature=abc"
    redirected = hydrate._HTTPSRedirect().redirect_request(request, None, 302, "Found", {}, target)
    assert redirected.full_url == target


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
    entry(data, "rnnoise")["precisions"]["a8w8"]["golden"].update(sha256=SHA, bytes=len(BODY))
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(_overlay(tmp_path, data)))
    assert {"rnnoise", "private-vad"} <= {r.id for r in zoo.records()}
    assert zoo.fetch("private-vad").read_bytes() == BODY
    resolved = zoo.resolve("zoo://private-vad/a8w8")
    assert resolved.model.read_bytes() == BODY and resolved.golden.read_bytes() == BODY
    assert server.hf[0][0] == "Example/models"


@pytest.mark.parametrize("location", ["local", "hf"])
def test_overlay_preserves_hf_only_private_source_boundary(monkeypatch, tmp_path, data, location):
    from helia_model_zoo import overlay

    root = _overlay(tmp_path, data)
    path = root / "models/private-vad/record.json"
    document = json.loads(path.read_text())
    document["precisions"]["a8w8"]["model"]["uri"] = HTTPS
    path.write_text(json.dumps(document))
    source = root
    if location == "hf":
        source = "hf://datasets/Example/private-records@" + "a" * 40
        monkeypatch.setattr(overlay, "_hf_records", lambda where, options: parse_records({"private-vad": document}))
    with pytest.raises(ManifestError, match="give every file as an hf:// URI"):
        zoo.load_overlay(source)


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
    resolved = zoo.resolve("zoo://rnnoise", root=root)
    assert (resolved.record, resolved.precision) == (item, item.precisions["a8w8"])
    assert (resolved.model, resolved.golden) == (
        root / "models/rnnoise/a8w8/model.tflite",
        root / "models/rnnoise/a8w8/golden.npz",
    )


@pytest.mark.parametrize(
    ("uri", "parts"),
    [
        ("zoo://rnnoise", ("rnnoise", None, None)),
        ("zoo://rnnoise/a8w8", ("rnnoise", "a8w8", None)),
        (f"zoo://rnnoise@{COMMIT}", ("rnnoise", None, COMMIT)),
        (f"zoo://dfnet2/a16w8@{COMMIT}", ("dfnet2", "a16w8", COMMIT)),
    ],
)
def test_reference_forms(uri, parts):
    reference = zoo.parse_reference(uri)
    assert (reference.id, reference.precision, reference.revision) == parts


@pytest.mark.parametrize(
    "uri",
    [
        "rnnoise",
        "zoo://",
        "zoo://rnnoise/",
        "zoo://RNNoise",
        "zoo://../rnnoise",
        "zoo://rnnoise/a8w8/model.tflite",
        "zoo://rnnoise@main",
        "zoo://rnnoise/a8w8@" + "0" * 39,
        "zoo://rnnoise/a8w8@" + "A" * 40,
    ],
)
def test_malformed_reference_is_refused(monkeypatch, uri):
    server = Server(monkeypatch)
    with pytest.raises(ValueError, match="expected zoo://"):
        zoo.resolve(uri)
    assert server.urls == []


def test_a_record_at_another_revision_comes_from_that_commit(monkeypatch, tmp_path, data):
    item = entry(data, "rnnoise")
    item["title"] = "RNNoise, as recorded at COMMIT"
    for key in ("model", "golden"):
        item["precisions"]["a8w8"][key].update(sha256=SHA, bytes=len(BODY))
    served = {"models/rnnoise/record.json": json.dumps(item).encode()}
    urls = []

    def download(url, destination, limit=None):
        urls.append(url)
        destination.write_bytes(served.get(url.split(f"/{COMMIT}/", 1)[1], BODY))

    monkeypatch.setattr(hydrate, "_download_url", download)
    monkeypatch.setattr(hydrate, "_install", lambda: {"vcs_info": {"commit_id": "f" * 40}})
    (tmp_path / "repo/models/rnnoise/a8w8").mkdir(parents=True)
    (tmp_path / "repo/models/rnnoise/a8w8/model.tflite").write_bytes(b"the checkout's other model")
    monkeypatch.setenv("HELIA_ZOO_ROOT", str(tmp_path / "repo"))

    resolved = zoo.resolve(f"zoo://rnnoise/a8w8@{COMMIT}")
    assert resolved.record.title == "RNNoise, as recorded at COMMIT" and resolved.record.revision == COMMIT
    assert resolved.model == tmp_path / "cache" / SHA / "model.tflite" and resolved.golden.read_bytes() == BODY
    assert urls[0] == f"https://raw.githubusercontent.com/AmbiqAI/helia-model-zoo/{COMMIT}/models/rnnoise/record.json"
    assert resolved.record.card() == tmp_path / "cache/text/repo" / COMMIT / "models/rnnoise/README.md"
    assert zoo.fetch("rnnoise", revision=COMMIT) == resolved.model
    assert resolved.record.card(revision="f" * 40) == resolved.record.card()
    assert all(f"/{COMMIT}/" in url for url in urls)
    assert zoo.get("rnnoise").revision is None
    other = zoo.resolve(f"zoo://rnnoise@{COMMIT}", cache=tmp_path / "other")
    assert (tmp_path / "other/text/repo" / COMMIT / "models/rnnoise/record.json").is_file()
    assert other.model == tmp_path / "other" / SHA / "model.tflite"
    assert zoo.fetch("rnnoise", revision=COMMIT, cache=tmp_path / "third") == tmp_path / "third" / SHA / "model.tflite"
    assert (tmp_path / "third/text/repo" / COMMIT / "models/rnnoise/record.json").is_file()
    with pytest.raises(TypeError, match="takes the revision in the reference"):
        zoo.resolve("zoo://rnnoise/a8w8", revision=COMMIT)


def test_an_explicit_revision_reads_only_hashed_files_from_an_explicit_checkout(monkeypatch, tmp_path):
    server = Server(monkeypatch)
    (tmp_path / "repo/audio/x").mkdir(parents=True)
    (tmp_path / "repo/audio/x/README.md").write_text("the checkout's card")
    (tmp_path / "repo/audio/x/model.tflite").write_bytes(BODY)
    monkeypatch.setenv("HELIA_ZOO_ROOT", str(tmp_path / "repo"))
    text, model = FileRef("repo://audio/x/README.md"), artifact("lfs://audio/x/model.tflite")
    assert fetch_file(text) == tmp_path / "repo/audio/x/README.md" and server.urls == []
    assert fetch_file(model, revision=COMMIT) == tmp_path / "cache" / SHA / "model.tflite" and len(server.urls) == 1
    assert fetch_file(model, root=tmp_path / "repo", revision=COMMIT) == tmp_path / "repo/audio/x/model.tflite"
    assert fetch_file(text, root=tmp_path / "repo", revision=COMMIT).read_bytes() == BODY and len(server.urls) == 2


@pytest.mark.parametrize("visibility", ["private", "public"])
def test_a_private_record_resolves_only_at_its_overlay_revision(monkeypatch, tmp_path, data, visibility):
    path = _overlay(tmp_path, data) / "models/private-vad/record.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), visibility=visibility)))
    record = path.read_bytes()
    server = Server(monkeypatch, record)
    server.files = ["models/private-vad/record.json"]
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", "hf://datasets/Example/index@" + "c" * 40)
    assert zoo.get("private-vad", "c" * 40).revision == "c" * 40
    with pytest.raises(KeyError, match="private-vad is private: set HELIA_ZOO_OVERLAY to its overlay dataset at d"):
        zoo.get("private-vad", "d" * 40)
    monkeypatch.setenv("HELIA_ZOO_OVERLAY", str(tmp_path / "overlay"))
    zoo.manifest.cache_clear()
    with pytest.raises(KeyError, match="is private"):
        zoo.get("private-vad", "c" * 40)
    assert server.urls == []


@pytest.mark.parametrize(("model_id", "revision"), [("rnnoise", "main"), ("rnnoise", "A" * 40), ("../x", COMMIT)])
def test_get_at_a_malformed_revision_or_id_is_refused(monkeypatch, model_id, revision):
    server = Server(monkeypatch)
    with pytest.raises(ValueError, match="full 40-hex commit"):
        zoo.get(model_id, revision)
    assert server.urls == []


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

    monkeypatch.setattr(hydrate._opener, "open", lambda url, timeout: Endless())
    with pytest.raises(FetchError, match="more than the expected 11 bytes"):
        fetch_file(artifact(LFS), revision=COMMIT)
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file()] == []


@pytest.mark.parametrize("error", [http.client.IncompleteRead(b"x"), ValueError("bad url")])
def test_transport_errors_become_fetch_errors(monkeypatch, error):
    def failing(url, timeout):
        raise error

    monkeypatch.setattr(hydrate._opener, "open", failing)
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
