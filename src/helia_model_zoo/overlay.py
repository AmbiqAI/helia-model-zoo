# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Overlay manifests: entries kept outside this public repository.

An overlay is located only by the user (``HELIA_ZOO_OVERLAY`` or an argument): a local path,
or an ``hf://`` URI pinned to a 40-hex revision. Nothing in this repository names one.
"""

from __future__ import annotations

import os
from pathlib import Path

from .manifest import FileRef, Manifest, ManifestError, load_manifest, parse_hf

OVERLAY_ENV = "HELIA_ZOO_OVERLAY"


def load_overlay(location: str | Path, **options) -> Manifest:
    """Load an overlay manifest@2 from a local path or an ``hf://`` URI; ``options`` go to ``fetch_file``.

    Raises:
        ManifestError: If the location or the document is invalid.
    """
    location = str(location)
    if location.startswith("hf://"):
        try:
            parse_hf(location)
        except ValueError as error:
            raise ManifestError(f"overlay: {error}") from None
        from .hydrate import fetch_file

        return load_manifest(fetch_file(FileRef(location), **options))
    if "://" in location:
        raise ManifestError(f"overlay: expected a local path or an hf:// URI, got {location!r}")
    if not Path(location).is_file():
        raise ManifestError(f"overlay: no such file: {location}")
    return load_manifest(Path(location))


def overlay_location() -> str | None:
    """The overlay named by ``HELIA_ZOO_OVERLAY``, if any."""
    return os.environ.get(OVERLAY_ENV) or None
