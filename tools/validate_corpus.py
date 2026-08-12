#!/usr/bin/env python3
"""Validate the model-zoo corpus manifest and its hydrated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from ai_edge_litert.interpreter import Interpreter


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"manifest path escapes corpus root: {relative_path}")
    return candidate


def _validate_artifact(root: Path, entry_id: str, path_value: str, expected_hash: str) -> Path:
    path = _resolve(root, path_value)
    if not path.is_file():
        raise ValueError(f"{entry_id}: required artifact is missing: {path_value}")
    if path.read_bytes()[: len(LFS_POINTER_PREFIX)] == LFS_POINTER_PREFIX:
        raise ValueError(f"{entry_id}: unresolved Git LFS pointer: {path_value}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"{entry_id}: SHA-256 mismatch for {path_value}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    return path


def _validate_tensor_keys(
    entry_id: str,
    arrays: np.lib.npyio.NpzFile,
    prefix: str,
    tensor_details: list[dict[str, Any]],
) -> None:
    expected_keys = [f"{prefix}_{index}" for index in range(len(tensor_details))]
    actual_keys = sorted(key for key in arrays.files if key.startswith(f"{prefix}_"))
    if actual_keys != expected_keys:
        raise ValueError(f"{entry_id}: expected {expected_keys}, found {actual_keys}")
    for key, detail in zip(expected_keys, tensor_details, strict=True):
        array = arrays[key]
        expected_shape = tuple(int(value) for value in detail["shape"])
        if array.shape != expected_shape:
            raise ValueError(
                f"{entry_id}: {key} shape mismatch: expected {expected_shape}, got {array.shape}"
            )
        expected_dtype = np.dtype(detail["dtype"])
        if array.dtype != expected_dtype:
            raise ValueError(
                f"{entry_id}: {key} dtype mismatch: expected {expected_dtype}, got {array.dtype}"
            )


def validate_manifest(manifest_path: Path) -> None:
    """Validate all entries and artifacts declared by a corpus manifest.

    Args:
        manifest_path: Path to the versioned JSON manifest.

    Raises:
        ValueError: If the manifest or an artifact is invalid.
    """
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported corpus manifest schema_version")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("corpus manifest must contain at least one entry")

    seen_ids: set[str] = set()
    for entry in entries:
        entry_id = entry["id"]
        if entry_id in seen_ids:
            raise ValueError(f"duplicate corpus ID: {entry_id}")
        seen_ids.add(entry_id)
        for metadata_key in (
            "reference_runtime",
            "reference_runtime_version",
            "provenance_reference",
            "license_reference",
        ):
            if not entry.get(metadata_key):
                raise ValueError(f"{entry_id}: missing {metadata_key}")
        for reference_key in ("provenance_reference", "license_reference"):
            reference_path = _resolve(root, entry[reference_key])
            if not reference_path.is_file():
                raise ValueError(f"{entry_id}: missing {reference_key}: {entry[reference_key]}")
        model_path = _validate_artifact(root, entry_id, entry["model"], entry["model_sha256"])
        golden_path = _validate_artifact(root, entry_id, entry["golden"], entry["golden_sha256"])

        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        with np.load(golden_path, allow_pickle=False) as arrays:
            allowed_keys = {
                *(f"input_{index}" for index in range(len(interpreter.get_input_details()))),
                *(f"output_{index}" for index in range(len(interpreter.get_output_details()))),
            }
            unexpected_keys = sorted(set(arrays.files) - allowed_keys)
            if unexpected_keys:
                raise ValueError(f"{entry_id}: unexpected NPZ keys: {unexpected_keys}")
            _validate_tensor_keys(entry_id, arrays, "input", interpreter.get_input_details())
            _validate_tensor_keys(entry_id, arrays, "output", interpreter.get_output_details())


def main() -> int:
    """Run corpus validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=Path("corpus-manifest-v1.json"))
    args = parser.parse_args()
    validate_manifest(args.manifest)
    print(f"validated corpus manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
