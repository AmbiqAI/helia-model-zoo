# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import json
import subprocess

import pytest
from conftest import entry

from helia_model_zoo.cli import main
from helia_model_zoo.guard import needles, scan
from helia_model_zoo.manifest import parse_manifest

SHA = "d" * 64
GOLDEN_SHA = "e" * 64
REPO = "Example/private-models"
HF = f"hf://datasets/{REPO}@{'a' * 40}/m/int8/model.tflite"
OVERLAY_AT = f"hf://datasets/Example/zoo-index@{'b' * 40}/manifest.json"


@pytest.fixture
def overlay(data):
    item = json.loads(json.dumps(entry(data, "rnnoise")))
    item.update(id="secret-enhancer", aliases={"secret-enhancer-v2": "int8"}, visibility="private")
    item["precisions"]["int8"]["model"].update(uri=HF, sha256=SHA)
    item["precisions"]["int8"]["golden"]["file"].update(sha256=GOLDEN_SHA)
    return parse_manifest({"schema": "helia-model-zoo/manifest@2", "entries": [item]})


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


def test_needles_cover_ids_aliases_repositories_and_hashes(overlay):
    found = needles(overlay, OVERLAY_AT)
    assert found == {
        "example/zoo-index": "overlay repository",
        "secret-enhancer": "ID of secret-enhancer",
        "secret-enhancer-v2": "alias of secret-enhancer",
        "example/private-models": "repository of secret-enhancer",
        SHA: "artifact sha256 of secret-enhancer",
        GOLDEN_SHA: "artifact sha256 of secret-enhancer",
    }


def test_names_shared_with_the_public_manifest_are_not_private(overlay, repo):
    # The overlay entry keeps RNNoise's public upstream sha256, which the public manifest also lists.
    rnnoise_upstream = overlay.entries[0].upstream.sha256
    assert rnnoise_upstream not in needles(overlay)
    commit(repo, "corpus.json", f'{{"model_sha256": "{rnnoise_upstream}"}}\n')
    assert scan(repo, overlay, base="base") == []


def test_clean_checkout_passes(repo, overlay):
    commit(repo, "notes.md", "nothing private here\n")
    assert scan(repo, overlay, location=OVERLAY_AT, base="base") == []


@pytest.mark.parametrize(
    "text",
    ["see Secret-Enhancer for details", "alias secret-enhancer-v2", f"oid sha256:{SHA}", "from EXAMPLE/zoo-index"],
)
def test_tracked_file_with_a_private_name_is_found(repo, overlay, text):
    commit(repo, "docs/page.md", f"intro\n{text}\n")
    hits = scan(repo, overlay, location=OVERLAY_AT, base="base")
    assert hits and all(hit.where == "docs/page.md:2" for hit in hits)


def test_untracked_file_is_scanned_and_ignored_file_is_not(repo, overlay):
    (repo / "draft.md").write_text("secret-enhancer\n")
    (repo / "ignored").mkdir()
    (repo / "ignored/local.md").write_text("secret-enhancer\n")
    assert [hit.where for hit in scan(repo, overlay, base="base")] == ["draft.md:1"]


def test_commit_messages_since_base_are_scanned(repo, overlay):
    git(repo, "commit", "-q", "--allow-empty", "-m", "feat: before base mentions secret-enhancer")
    git(repo, "branch", "-f", "base")
    git(repo, "commit", "-q", "--allow-empty", "-m", "feat: add model\n\nUses example/private-models.")
    hits = scan(repo, overlay, base="base")
    assert len(hits) == 1 and hits[0].where.startswith("commit ") and hits[0].kind == "repository of secret-enhancer"
    assert len(scan(repo, overlay, base=None)) == 2


def test_extra_text_is_scanned(repo, overlay, tmp_path):
    body = tmp_path / "pr-body.md"
    body.write_text("Summary\n\nAdds secret-enhancer.\n")
    assert [hit.where for hit in scan(repo, overlay, base="base", texts=(body,))] == [f"{body}:3"]


def test_cli_guard(repo, overlay, tmp_path, data, capsys):
    item = json.loads(json.dumps(entry(data, "rnnoise")))
    item.update(id="secret-enhancer", aliases={}, visibility="private")
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({"schema": "helia-model-zoo/manifest@2", "entries": [item]}))
    args = ["guard", "--overlay", str(path), "--root", str(repo), "--base", "base"]
    assert main(args) == 0
    commit(repo, "x.md", "secret-enhancer\n")
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "x.md:1: ID of secret-enhancer" in err and "do not publish" in err
    assert main([*args[:-1], "no-such-ref"]) == 2
