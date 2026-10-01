# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Model artifacts, golden fixtures and their manifest for Ambiq's helia tools."""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .manifest import (
    IO,
    Entry,
    FileRef,
    Golden,
    License,
    Manifest,
    ManifestError,
    Precision,
    StatePair,
    Tensor,
    Upstream,
    load_manifest,
    merge,
    parse_manifest,
)
from .overlay import load_overlay, overlay_location

__version__ = "0.1.0"


@cache
def manifest() -> Manifest:
    """The shipped manifest, plus the overlay that ``HELIA_ZOO_OVERLAY`` names, if any."""
    packaged = load_manifest()
    location = overlay_location()
    return packaged if location is None else merge(packaged, load_overlay(location))


def entries() -> tuple[Entry, ...]:
    """Every entry in :func:`manifest`."""
    return manifest().entries


def get(model_id: str) -> Entry:
    """The entry with this ID or alias."""
    return manifest().get(model_id)


def fetch(model_id: str, precision: str | None = None, **options) -> Path:
    """The verified local path of a model; an alias names its precision.

    ``options`` go to :func:`helia_model_zoo.hydrate.fetch_file`.
    """
    entry, aliased = manifest().resolve(model_id)
    if precision is None and aliased is not None:
        precision = aliased.name
    return entry.fetch(precision, **options)


def resolve(uri: str, **options) -> Path:
    """The verified local model path for ``zoo://<id>`` or ``zoo://<id>/<precision>``."""
    if not uri.startswith("zoo://") or not uri[len("zoo://") :].strip("/"):
        raise ValueError(f"expected zoo://<id>[/<precision>], got {uri!r}")
    model_id, _, precision = uri[len("zoo://") :].partition("/")
    return fetch(model_id, precision or None, **options)


def revision() -> str | None:
    """The repository commit this package's manifest describes (see ``hydrate.installed_revision``)."""
    from .hydrate import installed_revision

    return installed_revision()


__all__ = [
    "IO",
    "Entry",
    "FileRef",
    "Golden",
    "License",
    "Manifest",
    "ManifestError",
    "Precision",
    "StatePair",
    "Tensor",
    "Upstream",
    "entries",
    "fetch",
    "get",
    "load_manifest",
    "load_overlay",
    "manifest",
    "merge",
    "parse_manifest",
    "resolve",
    "revision",
]
