# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Refuse to publish a checkout of this public repository that names an overlay's private records.

Run before every push. The names to look for come from the overlay itself, so no list of
private names is ever committed here. What is searched, case-insensitively:

- every commit in ``base..HEAD``: its message, author and committer, the paths it adds or
  changes and the full content of those files, so a name added and later removed is still found;
- the index and the working tree: tracked and untracked (not ignored) files, their paths,
  symlink targets, and the current branch name;
- extra text files, such as a pull-request body.

What counts as a private name: each overlay record's ID and title (titles shorter
than six characters are skipped), the Hugging Face repositories it uses (``org/repo`` and
``repo``), its upstream repository, and every sha256 it pins, plus the overlay's own
repository. Names that the records at ``base`` also use are public and are skipped. Encoded
or split spellings are not detected.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .manifest import MODELS, FileRef, Manifest, ManifestError, Record, parse_hf, parse_records

_MIN_TITLE = 6


@dataclass(frozen=True)
class Hit:
    """A private name found at ``where`` (a file and line, a path, a commit, or a ref)."""

    where: str
    kind: str
    needle: str


def _files(record: Record) -> list[FileRef]:
    files = [record.card_file]
    for precision in record.precisions.values():
        files.append(precision.model)
        if precision.golden is not None:
            files.append(precision.golden.file)
    return files


def _names(record: Record) -> dict[str, str]:
    found: dict[str, str] = {record.id.lower(): f"ID of {record.id}"}
    if len(record.title) >= _MIN_TITLE:
        found[record.title.lower()] = f"title of {record.id}"
    for ref in _files(record):
        if ref.scheme == "hf":
            repo = parse_hf(ref.uri).repo_id.lower()
            found[repo] = f"repository of {record.id}"
            found[repo.split("/", 1)[1]] = f"repository of {record.id}"
        if ref.sha256:
            found[ref.sha256] = f"sha256 pinned by {record.id}"
    if record.upstream:
        found[record.upstream.repo.lower().removesuffix("/")] = f"upstream of {record.id}"
        if record.upstream.sha256:
            found[record.upstream.sha256] = f"sha256 pinned by {record.id}"
    return found


def needles(overlay: Manifest, location: str | None = None, public: Manifest | None = None) -> dict[str, str]:
    """Every private name in ``overlay``, lowercased, mapped to what it is.

    Names that ``public`` also uses are left out: a private record may share a public
    upstream or artifact.
    """
    found: dict[str, str] = {}
    if location and location.startswith("hf://"):
        repo = parse_hf(location.rstrip("/") + "/.").repo_id.lower()
        found[repo] = found[repo.split("/", 1)[1]] = "overlay repository"
    for record in overlay.records:
        found |= _names(record)
    shared = set() if public is None else {n for r in public.records for n in _names(r)}
    return {needle: kind for needle, kind in found.items() if needle not in shared}


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=not binary)
    return result.stdout


def public_manifest_at(root: Path, base: str | None) -> Manifest | None:
    """The records committed at ``base``, or None if there are none.

    Raises:
        ManifestError: If a record at ``base`` does not parse.
    """
    if base is None:
        return None
    try:
        listing = _git(root, "ls-tree", "-r", "--name-only", base, "--", MODELS)
    except subprocess.CalledProcessError:
        return None
    documents = {}
    for name in listing.splitlines():
        parts = name.split("/")
        if len(parts) == 3 and parts[2] == "record.json":
            documents[parts[1]] = json.loads(_git(root, "show", f"{base}:{name}"))
    if not documents:
        return None
    try:
        return parse_records(documents)
    except ValueError as error:
        raise ManifestError(f"records at {base}: {error}") from None


class _Scanner:
    def __init__(self, lookup: dict[str, str]):
        self.lookup, self.hits = lookup, []

    def text(self, text: str, where: str, lines: bool = True) -> None:
        for number, line in enumerate(text.lower().splitlines() or [""], 1):
            for needle, kind in self.lookup.items():
                if needle in line:
                    self.hits.append(Hit(f"{where}:{number}" if lines else where, kind, needle))

    def data(self, data: bytes, where: str) -> None:
        self.text(data.decode("utf-8", errors="replace"), where)


def _blobs(root: Path, ids: list[str]) -> dict[str, bytes]:
    """Read many blobs at once with ``git cat-file --batch``."""
    if not ids:
        return {}
    unique = list(dict.fromkeys(ids))
    out = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="".join(f"{i}\n" for i in unique).encode(),
        check=True,
        capture_output=True,
    ).stdout
    blobs, offset = {}, 0
    for blob in unique:
        header_end = out.index(b"\n", offset)
        header = out[offset:header_end].split(b" ")
        if len(header) != 3:  # "<id> missing", e.g. a submodule commit: no content here to scan
            blobs[blob], offset = b"", header_end + 1
            continue
        name, kind, size = header
        start = header_end + 1
        blobs[blob] = out[start : start + int(size)] if kind == b"blob" else b""
        offset = start + int(size) + 1
    return blobs


def scan(
    root: Path,
    overlay: Manifest,
    *,
    location: str | None = None,
    base: str | None = "origin/main",
    texts: tuple[Path, ...] = (),
) -> list[Hit]:
    """Search a checkout and the commits it would publish for the overlay's private names.

    See the module docstring for what is searched. ``base`` None searches all of HEAD's history.

    Raises:
        subprocess.CalledProcessError: If git fails, for example on an unknown ``base``.
        ManifestError: If the manifest at ``base`` does not parse.
    """
    root = Path(root)
    scanner = _Scanner(needles(overlay, location, public_manifest_at(root, base)))
    revisions = _git(root, "rev-list", f"{base}..HEAD" if base else "HEAD").split()
    for commit in revisions:
        header = _git(root, "show", "-s", "--format=%an <%ae>%n%cn <%ce>%n%B", commit)
        scanner.text(header, f"commit {commit[:12]}")
        # -m: a merge is compared with each parent, so content added while resolving it is seen.
        raw = _git(root, "diff-tree", "-r", "-m", "-z", "--no-commit-id", "--root", "--no-renames", commit, binary=True)
        fields = raw.split(b"\0")
        changes = [
            (fields[i].split(b" ")[3].decode(), fields[i + 1].decode(errors="replace"))
            for i in range(0, len(fields) - 1, 2)
        ]
        added = [(blob, path) for blob, path in changes if set(blob) != {"0"}]
        for blob, path in changes:
            scanner.text(path, f"commit {commit[:12]} path {path}", lines=False)
        contents = _blobs(root, [blob for blob, _ in added])
        for blob, path in added:
            scanner.data(contents[blob], f"commit {commit[:12]} {path}")
    staged = _git(root, "ls-files", "-z", "--stage", binary=True).split(b"\0")
    # Every stage of every path, so each side of an unresolved conflict is read.
    entries = [line.decode(errors="replace").split("\t", 1) for line in staged if line]
    index = [(path, meta.split(" ")[1]) for meta, path in entries]
    contents = _blobs(root, [blob for _, blob in index])
    for path, blob in index:
        scanner.data(contents[blob], f"index {path}")
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", binary=True)
    # os.fsdecode keeps undecodable bytes (surrogateescape), so the file can still be opened.
    for name in sorted({os.fsdecode(n) for n in listing.split(b"\0") if n}):
        scanner.text(name, f"path {name}", lines=False)
        path = root / name
        if path.is_symlink():
            scanner.text(os.readlink(path), f"symlink {name}", lines=False)
        elif path.is_file():
            scanner.data(path.read_bytes(), name)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    scanner.text(branch, f"branch {branch}", lines=False)
    for text in texts:
        scanner.data(Path(text).read_bytes(), str(text))
    return sorted(set(scanner.hits), key=lambda hit: (hit.where, hit.needle))
