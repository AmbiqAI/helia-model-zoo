# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Overlays: records kept outside this public repository.

An overlay has this repository's layout (``models/<id>/record.json``) and is located only by the
user (``HELIA_ZOO_OVERLAY`` or an argument): a local directory, or the root of a Hugging Face
dataset as ``hf://datasets/<org>/<repo>@<40-hex revision>/``. Nothing in this repository names one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .manifest import MODELS, FileRef, Manifest, ManifestError, load_manifest, parse_hf, parse_records

OVERLAY_ENV = "HELIA_ZOO_OVERLAY"


def _hf_records(location: str, options: dict) -> Manifest:
    from .hydrate import FetchError, fetch_file

    try:
        where = parse_hf(location.rstrip("/") + "/.")
    except ValueError as error:
        raise ManifestError(f"overlay: {error}") from None
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ManifestError("an hf:// overlay needs huggingface_hub: install helia-model-zoo[hf]") from error
    prefix = location.rstrip("/").split("@", 1)[0]
    files = HfApi().list_repo_files(where.repo_id, repo_type=where.repo_type, revision=where.revision)
    documents = {}
    for name in files:
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == MODELS and parts[2] == "record.json":
            try:
                path = fetch_file(FileRef(f"{prefix}@{where.revision}/{name}"), **options)
            except FetchError as error:
                raise ManifestError(f"overlay: {error}") from None
            documents[parts[1]] = json.loads(Path(path).read_text(encoding="utf-8"))
    if not documents:
        raise ManifestError(f"overlay: no {MODELS}/<id>/record.json in {location}")
    return parse_records(documents)


def load_overlay(location: str | Path, **options) -> Manifest:
    """Load an overlay's records from a local directory or an ``hf://`` dataset root.

    ``options`` go to ``fetch_file`` for an ``hf://`` overlay.

    Raises:
        ManifestError: If the location or a record is invalid.
    """
    location = str(location)
    if location.startswith("hf://"):
        return _hf_records(location, options)
    if "://" in location:
        raise ManifestError(f"overlay: expected a local directory or an hf:// dataset root, got {location!r}")
    if not (Path(location) / MODELS).is_dir():
        raise ManifestError(f"overlay: no {MODELS}/ directory in {location}")
    return load_manifest(Path(location) / MODELS)


def overlay_location() -> str | None:
    """The overlay named by ``HELIA_ZOO_OVERLAY``, if any."""
    return os.environ.get(OVERLAY_ENV) or None
