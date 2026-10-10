# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Overlays: records kept outside this public repository.

An overlay has this repository's layout (``models/<id>/record.json``) and is located only by the
user (``HELIA_ZOO_OVERLAY`` or an argument): a local directory, or the root of a Hugging Face
dataset as ``hf://datasets/<org>/<repo>@<40-hex revision>/``. Nothing in this repository names one.
Its records give every file as an ``hf://`` URI: a ``path`` would resolve against this public
repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from .manifest import (
    MODELS,
    FileRef,
    HfLocation,
    Manifest,
    ManifestError,
    load_manifest,
    parse_hf,
    parse_records,
    read_record,
)

OVERLAY_ENV = "HELIA_ZOO_OVERLAY"


def hf_root(location: str) -> HfLocation:
    """The dataset an ``hf://datasets/<org>/<repo>@<40-hex revision>/`` overlay root names.

    Raises:
        ManifestError: If ``location`` is not such a root.
    """
    try:
        where = parse_hf(f"{location.rstrip('/')}/{MODELS}")
    except ValueError:
        where = None
    if where is None or where.repo_type != "dataset" or where.path != MODELS:
        raise ManifestError(
            f"overlay: expected hf://datasets/<org>/<repo>@<40-hex commit>/ as an overlay root, got {location!r}"
        )
    return where


def _hf_records(location: str, options: dict) -> Manifest:
    from .hydrate import FetchError, fetch_file

    where = hf_root(location)
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ManifestError("an hf:// overlay needs huggingface_hub: install helia-model-zoo[hf]") from error
    files = HfApi().list_repo_files(
        where.repo_id,
        repo_type=where.repo_type,
        revision=where.revision,
        token=False if options.get("anonymous") else None,
    )
    documents = {}
    for name in files:
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == MODELS and parts[2] == "record.json":
            try:
                path = fetch_file(FileRef(f"hf://datasets/{where.repo_id}@{where.revision}/{name}"), **options)
            except (FetchError, ValueError) as error:
                raise ManifestError(f"overlay: {error}") from None
            documents[parts[1]] = read_record(Path(path).read_bytes(), name)
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
        overlay = _hf_records(location, options)
    elif "://" in location:
        raise ManifestError(f"overlay: expected a local directory or an hf:// dataset root, got {location!r}")
    elif not (Path(location) / MODELS).is_dir():
        raise ManifestError(f"overlay: no {MODELS}/ directory in {location}")
    else:
        overlay = load_manifest(Path(location) / MODELS)
    for record in overlay.records:
        if any(ref.scheme != "hf" for ref in record.files()):
            raise ManifestError(f"overlay record {record.id}: give every file as an hf:// URI, not a path")
    return overlay


def overlay_location() -> str | None:
    """The overlay named by ``HELIA_ZOO_OVERLAY``, if any."""
    return os.environ.get(OVERLAY_ENV) or None
