# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Model artifacts, golden fixtures and their records for Ambiq's helia tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

from .manifest import (
    _ID,
    _REVISION,
    IO,
    MODELS,
    FileRef,
    Golden,
    Manifest,
    ManifestError,
    Precision,
    Record,
    StatePair,
    Tensor,
    Upstream,
    load_manifest,
    merge,
    parse_record,
    parse_records,
    read_record,
)
from .overlay import OVERLAY_ENV, hf_root, load_overlay, overlay_location

__version__ = "0.1.0"

_REFERENCE = re.compile(
    rf"zoo://(?P<id>{_ID.pattern})(?:/(?P<precision>[a-z0-9]+))?(?:@(?P<revision>{_REVISION.pattern}))?"
)


@cache
def manifest() -> Manifest:
    """The shipped records, plus the overlay that ``HELIA_ZOO_OVERLAY`` names, if any.

    The result is cached for the process; call ``manifest.cache_clear()`` after changing the variable.
    """
    packaged = load_manifest()
    location = overlay_location()
    return packaged if location is None else merge(packaged, load_overlay(location))


def records() -> tuple[Record, ...]:
    """Every record in :func:`manifest`."""
    return manifest().records


def get(model_id: str, revision: str | None = None, **options) -> Record:
    """The record with this ID: the installed one, or the one at ``revision``, a full 40-hex commit.

    At a revision, a public record is downloaded from this repository at that commit (``options`` go to
    :func:`helia_model_zoo.hydrate.fetch_file`), and its files then come from the same commit. Only the overlay
    makes an ID private; a private record is the overlay's, which must be pinned to that revision.

    Raises:
        KeyError: If the ID is not installed (without a revision), or is private and its overlay is at another
            revision.
        ValueError: If ``revision`` is not a full 40-hex commit.
        helia_model_zoo.hydrate.FetchError: If the record cannot be downloaded at that revision.
        ManifestError: If the record is not record@1.
    """
    if revision is None:
        return manifest().get(model_id)
    if not _REVISION.fullmatch(revision) or not _ID.fullmatch(model_id):
        raise ValueError(f"expected a model ID and a full 40-hex commit, got {model_id!r} at {revision!r}")
    try:
        installed = manifest().get(model_id)
    except KeyError:
        installed = None
    if installed is not None and installed.visibility == "private":
        location = overlay_location()
        pinned = hf_root(location).revision if location and location.startswith("hf://") else None
        if pinned != revision:
            raise KeyError(f"{model_id} is private: set {OVERLAY_ENV} to its overlay dataset at {revision}")
        return replace(installed, revision=revision)
    from .hydrate import fetch_file

    path = fetch_file(FileRef(f"repo://{MODELS}/{model_id}/record.json"), **options, revision=revision)
    return replace(parse_record(read_record(path.read_bytes(), path), model_id), revision=revision)


def fetch(model_id: str, precision: str | None = None, revision: str | None = None, **options) -> Path:
    """The verified local path of a model, as recorded at ``revision`` (see :func:`get`).

    ``options`` go to :func:`helia_model_zoo.hydrate.fetch_file`.
    """
    return get(model_id, revision, **options).fetch(precision, **options)


@dataclass(frozen=True)
class Reference:
    """A parsed ``zoo://<id>[/<precision>][@<revision>]``; ``revision`` is None when unpinned."""

    id: str
    precision: str | None
    revision: str | None


def parse_reference(uri: str) -> Reference:
    """Split a ``zoo://`` reference; a revision must be a full 40-hex commit.

    Raises:
        ValueError: If ``uri`` is not of that form.
    """
    match = _REFERENCE.fullmatch(uri)
    if match is None:
        raise ValueError(f"expected zoo://<id>[/<precision>][@<40-hex commit>], got {uri!r}")
    return Reference(match["id"], match["precision"], match["revision"])


@dataclass(frozen=True)
class Resolved:
    """A resolved reference: its record and precision, and their verified local files."""

    record: Record
    precision: Precision
    model: Path
    golden: Path | None


def resolve(uri: str, **options) -> Resolved:
    """Resolve ``zoo://<id>[/<precision>][@<revision>]`` (see :func:`get`), fetching its model and golden.

    Without a revision, the installed record is used. ``options`` go to :func:`helia_model_zoo.hydrate.fetch_file`.
    """
    reference = parse_reference(uri)
    record = get(reference.id, reference.revision, **options)
    precision = record.precision(reference.precision)
    golden = None if precision.golden is None else record.fetch_file(precision.golden.file, **options)
    return Resolved(record, precision, record.fetch_file(precision.model, **options), golden)


def revision() -> str | None:
    """The repository commit this package's records describe (see ``hydrate.installed_revision``)."""
    from .hydrate import installed_revision

    return installed_revision()


__all__ = [
    "IO",
    "FileRef",
    "Golden",
    "Manifest",
    "ManifestError",
    "Precision",
    "Record",
    "Reference",
    "Resolved",
    "StatePair",
    "Tensor",
    "Upstream",
    "fetch",
    "get",
    "load_manifest",
    "load_overlay",
    "manifest",
    "merge",
    "parse_record",
    "parse_reference",
    "parse_records",
    "records",
    "resolve",
    "revision",
]
