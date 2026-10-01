# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import json
import subprocess

import pytest
from conftest import PACKAGED, entry

from helia_model_zoo.cli import main
from helia_model_zoo.guard import MANIFEST_PATH, needles, scan
from helia_model_zoo.manifest import parse_manifest

SHA = "d" * 64
GOLDEN_SHA = "e" * 64
UPSTREAM_SHA = "c" * 64
REPO = "Example/private-models"
HF = f"hf://datasets/{REPO}@{'a' * 40}/m/int8/model.tflite"
CARD = f"hf://datasets/Example/private-cards@{'a' * 40}/m/README.md"
OVERLAY_AT = f"hf://datasets/Example/zoo-index@{'b' * 40}/manifest.json"


def overlay_item(data):
    item = json.loads(json.dumps(entry(data, "rnnoise")))
    item.update(id="secret-enhancer", aliases={"secret-enhancer-v2": "int8"}, visibility="private", title="Hush Pro")
    item["card"] = {"uri": CARD}
    item["upstream"] = {
        "repo": "https://example.com/vendor/hush",
        "revision": None,
        "path": None,
        "sha256": UPSTREAM_SHA,
    }
    item["precisions"]["int8"]["model"].update(uri=HF, sha256=SHA)
    item["precisions"]["int8"]["golden"]["file"].update(sha256=GOLDEN_SHA)
    return item


@pytest.fixture
def overlay(data):
    return parse_manifest({"schema": "helia-model-zoo/manifest@2", "entries": [overlay_item(data)]})


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "test")
    (root / "README.md").write_text("public models\n")
    (root / ".gitignore").write_text("ignored/\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "docs: start")
    git(root, "branch", "base")
    return root


def commit(root, name, text, message="docs: change"):
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)
    git(root, "add", name)
    git(root, "commit", "-q", "-m", message)


def wheres(root, overlay, **options):
    return sorted({hit.where for hit in scan(root, overlay, base="base", **options)})


def test_needles(overlay):
    assert needles(overlay, OVERLAY_AT) == {
        "example/zoo-index": "overlay repository",
        "zoo-index": "overlay repository",
        "secret-enhancer": "ID of secret-enhancer",
        "secret-enhancer-v2": "alias of secret-enhancer",
        "hush pro": "title of secret-enhancer",
        "example/private-cards": "repository of secret-enhancer",
        "private-cards": "repository of secret-enhancer",
        "example/private-models": "repository of secret-enhancer",
        "private-models": "repository of secret-enhancer",
        "https://example.com/vendor/hush": "upstream of secret-enhancer",
        SHA: "sha256 pinned by secret-enhancer",
        GOLDEN_SHA: "sha256 pinned by secret-enhancer",
        UPSTREAM_SHA: "sha256 pinned by secret-enhancer",
    }


def test_short_titles_are_not_needles(data):
    item = overlay_item(data)
    item["title"] = "VAD"
    assert "vad" not in needles(parse_manifest({"schema": "helia-model-zoo/manifest@2", "entries": [item]}))


def test_clean_checkout_passes(repo, overlay):
    commit(repo, "notes.md", "nothing private here\n")
    assert scan(repo, overlay, location=OVERLAY_AT, base="base") == []


@pytest.mark.parametrize(
    "text",
    [
        "see Secret-Enhancer for details",
        "alias secret-enhancer-v2",
        f"oid sha256:{SHA}",
        "from EXAMPLE/zoo-index",
        "the Hush Pro model",
        "card at private-cards",
        UPSTREAM_SHA,
        "https://example.com/vendor/hush/archive",
    ],
)
def test_committed_name_is_found_in_history_index_and_tree(repo, overlay, text):
    commit(repo, "docs/page.md", f"intro\n{text}\n")
    found = wheres(repo, overlay, location=OVERLAY_AT)
    assert len(found) == 3
    assert found[0].startswith("commit ") and found[0].endswith(" docs/page.md:2")
    assert found[1:] == ["docs/page.md:2", "index docs/page.md:2"]


def test_name_removed_by_a_later_commit_is_still_found(repo, overlay):
    commit(repo, "docs/page.md", "uses secret-enhancer\n")
    commit(repo, "docs/page.md", "uses nothing\n")
    found = wheres(repo, overlay)
    assert len(found) == 1 and found[0].endswith(" docs/page.md:1")


def test_binary_content_in_history_is_scanned(repo, overlay):
    (repo / "model.bin").write_bytes(b"\x00\x01TFL3secret-enhancer\x00")
    git(repo, "add", "model.bin")
    git(repo, "commit", "-q", "-m", "feat: add model")
    git(repo, "rm", "-q", "model.bin")
    git(repo, "commit", "-q", "-m", "fix: remove model")
    assert any(w.endswith(" model.bin:1") for w in wheres(repo, overlay))


def test_staged_name_is_found_when_the_working_file_is_clean(repo, overlay):
    (repo / "page.md").write_text("secret-enhancer\n")
    git(repo, "add", "page.md")
    (repo / "page.md").write_text("clean\n")
    assert wheres(repo, overlay) == ["index page.md:1"]


def test_file_name_in_history_is_found(repo, overlay):
    commit(repo, "secret-enhancer/README.md", "harmless\n")
    found = wheres(repo, overlay)
    assert any(w.startswith("commit ") and w.endswith("path secret-enhancer/README.md") for w in found)
    assert "path secret-enhancer/README.md" in found


def test_symlink_target_is_found(repo, overlay):
    (repo / "link").symlink_to("models/secret-enhancer.tflite")
    assert wheres(repo, overlay) == ["symlink link"]


def test_branch_name_is_found(repo, overlay):
    git(repo, "checkout", "-q", "-b", "add-secret-enhancer")
    assert wheres(repo, overlay) == ["branch add-secret-enhancer"]


def test_author_is_found(repo, overlay):
    git(repo, "-c", "user.name=Hush Pro team", "commit", "-q", "--allow-empty", "-m", "chore: empty")
    # Author on line 1, committer on line 2 of the commit header.
    found = wheres(repo, overlay)
    assert len(found) == 2 and found[0].endswith(":1") and found[1].endswith(":2")


def test_untracked_file_is_scanned_and_ignored_file_is_not(repo, overlay):
    (repo / "draft.md").write_text("secret-enhancer\n")
    (repo / "ignored").mkdir()
    (repo / "ignored/local.md").write_text("secret-enhancer\n")
    assert wheres(repo, overlay) == ["draft.md:1"]


def test_commit_messages_since_base_are_scanned(repo, overlay):
    git(repo, "commit", "-q", "--allow-empty", "-m", "feat: before base mentions secret-enhancer")
    git(repo, "branch", "-f", "base")
    git(repo, "commit", "-q", "--allow-empty", "-m", "feat: add model\n\nUses example/private-models.")
    hits = scan(repo, overlay, base="base")
    assert {hit.kind for hit in hits} == {"repository of secret-enhancer"}
    assert all(hit.where.startswith("commit ") for hit in hits)
    assert len({hit.where for hit in scan(repo, overlay, base=None)}) == 2


def test_extra_text_is_scanned(repo, overlay, tmp_path):
    body = tmp_path / "pr-body.md"
    body.write_text("Summary\n\nAdds secret-enhancer.\n")
    assert wheres(repo, overlay, texts=(body,)) == [f"{body}:3"]


def commit_manifest(repo, document):
    commit(repo, MANIFEST_PATH, json.dumps(document), "feat: manifest")


def test_names_public_at_base_are_skipped(repo, overlay, data):
    public = json.loads(json.dumps(PACKAGED))
    public_item = overlay_item(data)
    public_item.update(id="hush-pro", aliases={}, visibility="public")
    public["entries"].append(public_item)
    commit_manifest(repo, public)
    git(repo, "branch", "-f", "base")
    commit(repo, "docs/page.md", f"Hush Pro, {UPSTREAM_SHA}, example/private-models\n")
    assert scan(repo, overlay, base="base") == []
    commit(repo, "docs/other.md", "secret-enhancer\n")
    assert scan(repo, overlay, base="base")


def test_entry_copied_into_the_branch_manifest_is_found(repo, overlay, data):
    commit_manifest(repo, PACKAGED)
    git(repo, "branch", "-f", "base")
    leaked = json.loads(json.dumps(PACKAGED))
    leaked["entries"].append(overlay_item(data) | {"visibility": "public"})
    commit_manifest(repo, leaked)
    found = wheres(repo, overlay)
    assert any(w.startswith("commit ") and MANIFEST_PATH in w for w in found)
    assert any(w.startswith(MANIFEST_PATH) for w in found)


def write_overlay(tmp_path, data):
    item = overlay_item(data)
    item["aliases"] = {}
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": "helia-model-zoo/manifest@2", "entries": [item]}))
    return path


def test_cli_guard(repo, tmp_path, data, capsys):
    args = ["guard", "--overlay", str(write_overlay(tmp_path, data)), "--root", str(repo), "--base", "base"]
    assert main(args) == 0
    commit(repo, "x.md", "secret-enhancer\n")
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "x.md:1: ID of secret-enhancer" in err and "do not publish" in err
    assert main([*args[:-1], "no-such-ref"]) == 2
    assert main([*args[:2], str(tmp_path / "missing.json"), *args[3:]]) == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_guard_defaults_to_origin_main(repo, tmp_path, data):
    commit(repo, "x.md", "secret-enhancer\n")
    git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit(repo, "x.md", "clean\n")
    overlay = str(write_overlay(tmp_path, data))
    # The name is only in a commit before origin/main, which this push does not publish.
    assert main(["guard", "--overlay", overlay, "--root", str(repo)]) == 0
    git(repo, "update-ref", "refs/remotes/origin/main", "base")
    assert main(["guard", "--overlay", overlay, "--root", str(repo)]) == 1


def test_name_added_while_resolving_a_merge_is_found(repo, overlay):
    commit(repo, "a.md", "one\n")
    git(repo, "checkout", "-q", "-b", "side", "base")
    commit(repo, "b.md", "two\n")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "--no-commit", "side")
    (repo / "b.md").write_text("two, via secret-enhancer\n")
    git(repo, "add", "b.md")
    git(repo, "commit", "-q", "-m", "Merge branch side")
    commit(repo, "b.md", "two\n")
    found = wheres(repo, overlay)
    assert len(found) == 1 and found[0].startswith("commit ") and found[0].endswith(" b.md:1")


def test_submodule_and_odd_file_names_do_not_stop_the_scan(repo, overlay):
    sub = repo.parent / "sub"
    sub.mkdir()
    git(sub, "init", "-q")
    git(sub, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    head = subprocess.run(["git", "-C", str(sub), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},vendor")
    git(repo, "commit", "-q", "-m", "chore: add submodule")
    (repo / b"caf\xe9-secret-enhancer.md".decode("utf-8", "surrogateescape")).write_text("x\n")
    found = wheres(repo, overlay)
    assert any("secret-enhancer.md" in w and w.startswith("path ") for w in found)


def test_conflicted_index_is_scanned(repo, overlay):
    commit(repo, "c.md", "base\n")
    git(repo, "branch", "-f", "base")
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, "c.md", "theirs mention secret-enhancer\n")
    git(repo, "checkout", "-q", "main")
    commit(repo, "c.md", "ours\n")
    subprocess.run(["git", "-C", str(repo), "merge", "-q", "side"], capture_output=True)
    (repo / "c.md").write_text("resolved but not staged\n")
    assert any(w.startswith("index c.md") for w in wheres(repo, overlay))


def test_untracked_file_with_a_non_utf8_name_is_read(repo, overlay):
    (repo / b"draft-\xff.md".decode("utf-8", "surrogateescape")).write_text("secret-enhancer\n")
    assert any(w.startswith("draft-") and w.endswith(".md:1") for w in wheres(repo, overlay))
