# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Model artifacts, golden fixtures and their manifest for Ambiq's helia tools."""

from __future__ import annotations

from functools import cache

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
    parse_manifest,
)

__version__ = "0.1.0"


@cache
def manifest() -> Manifest:
    """The manifest shipped with this package."""
    return load_manifest()


def entries() -> tuple[Entry, ...]:
    """Every entry in the shipped manifest."""
    return manifest().entries


def get(model_id: str) -> Entry:
    """The entry with this ID or v1 alias."""
    return manifest().get(model_id)


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
    "get",
    "load_manifest",
    "manifest",
    "parse_manifest",
]
