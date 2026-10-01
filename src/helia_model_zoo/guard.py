# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Find any trace of an overlay's private entries in a checkout of this public repository.

Run before every push: the needles come from the overlay itself, so no list of private names
is ever committed here.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .manifest import Entry, FileRef, Manifest, load_manifest, parse_hf


@dataclass(frozen=True)
class Hit:
    """A private name found at ``where`` (a file and line, or a commit)."""

    where: str
    kind: str
    needle: str


def _files(entry: Entry) -> list[FileRef]:
    files = [entry.card_file, entry.license.reference]
    for precision in entry.precisions.values():
        files.append(precision.model)
        if precision.golden is not None:
            files.append(precision.golden.file)
    return files


def needles(overlay: Manifest, location: str | None = None, public: Manifest | None = None) -> dict[str, str]:
    """Every private name in ``overlay``, lowercased, mapped to what it is.

    Names that ``public`` (by default the packaged manifest) also uses are left out: a private
    entry may share a public upstream or artifact, and those are not private.
    """
    found: dict[str, str] = {}
    if location and location.startswith("hf://"):
        found[parse_hf(location).repo_id.lower()] = "overlay repository"
    for entry in overlay.entries:
        found[entry.id.lower()] = f"ID of {entry.id}"
        for alias in entry.aliases:
            found[alias.lower()] = f"alias of {entry.id}"
        for ref in _files(entry):
            if ref.scheme == "hf":
                found[parse_hf(ref.uri).repo_id.lower()] = f"repository of {entry.id}"
            if ref.sha256:
                found[ref.sha256] = f"artifact sha256 of {entry.id}"
        if entry.upstream and entry.upstream.sha256:
            found[entry.upstream.sha256] = f"upstream sha256 of {entry.id}"
    public = load_manifest() if public is None else public
    shared = set(needles_of_public(public))
    return {needle: kind for needle, kind in found.items() if needle not in shared}


def needles_of_public(manifest: Manifest) -> set[str]:
    """The same kinds of names, for a public manifest."""
    names: set[str] = set()
    for entry in manifest.entries:
        names |= {entry.id.lower(), *(alias.lower() for alias in entry.aliases)}
        for ref in _files(entry):
            if ref.scheme == "hf":
                names.add(parse_hf(ref.uri).repo_id.lower())
            if ref.sha256:
                names.add(ref.sha256)
        if entry.upstream and entry.upstream.sha256:
            names.add(entry.upstream.sha256)
    return names


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _scan(text: str, where: str, lookup: dict[str, str], hits: list[Hit]) -> None:
    for number, line in enumerate(text.lower().splitlines(), 1):
        for needle, kind in lookup.items():
            if needle in line:
                hits.append(Hit(f"{where}:{number}", kind, needle))


def scan(
    root: Path,
    overlay: Manifest,
    *,
    location: str | None = None,
    base: str | None = "origin/main",
    texts: tuple[Path, ...] = (),
) -> list[Hit]:
    """Search a checkout for the overlay's private names.

    Searches every tracked and untracked (not ignored) file, the messages of commits in
    ``base..HEAD`` (all of HEAD's history when ``base`` is None), and ``texts`` such as a PR body.
    Matching is case-insensitive.
    """
    lookup = needles(overlay, location)
    hits: list[Hit] = []
    listing = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    for name in sorted(set(filter(None, listing.split("\0")))):
        path = Path(root) / name
        if path.is_file() and not path.is_symlink():
            _scan(path.read_bytes().decode("utf-8", errors="replace"), name, lookup, hits)
    log = _git(root, "log", "--format=%H%x00%B%x01", base + "..HEAD" if base else "HEAD")
    for record in filter(None, (r.strip() for r in log.split("\x01"))):
        commit, _, message = record.partition("\0")
        _scan(message, f"commit {commit[:12]}", lookup, hits)
    for text in texts:
        _scan(Path(text).read_text(encoding="utf-8", errors="replace"), str(text), lookup, hits)
    return hits
