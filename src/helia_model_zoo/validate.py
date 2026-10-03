# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Validate a manifest@2 against a hydrated checkout of this repository."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np

from . import golden, runtime
from .hydrate import LFS_POINTER_PREFIX, FetchError, fetch_file, sha256_file
from .manifest import FileRef, Manifest, Precision, Record, load_manifest

__all__ = ["ValidationError", "sha256_file", "validate"]


class ValidationError(ValueError):
    """One or more problems were found; ``problems`` lists them all."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def _check_file(
    root: Path, ref: FileRef, where: str, problems: list[str], cache: Path | None = None, anonymous: bool = True
) -> Path | None:
    """Check that a file exists and, for an artifact, that its bytes are hydrated and match.

    A file outside this repository is fetched, without credentials when ``anonymous``.
    """
    if not ref.in_repository:
        try:
            return fetch_file(ref, cache=cache, anonymous=anonymous)
        except FetchError as error:
            note = " (fetched without credentials, as a public record must be)" if anonymous else ""
            problems.append(f"{where}: {error}{note}")
            return None
    try:
        path = ref.resolve(root)
    except ValueError as error:
        problems.append(f"{where}: {error}")
        return None
    if not path.is_file():
        problems.append(f"{where}: missing file {ref.uri}")
        return None
    if ref.sha256 is None:
        return path
    with path.open("rb") as stream:
        if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            problems.append(f"{where}: unresolved Git LFS pointer {ref.uri}")
            return None
    if ref.bytes is not None and path.stat().st_size != ref.bytes:
        problems.append(f"{where}: {ref.uri} is {path.stat().st_size} bytes, expected {ref.bytes}")
        return None
    actual = sha256_file(path)
    if actual != ref.sha256:
        problems.append(f"{where}: sha256 mismatch for {ref.uri}: expected {ref.sha256}, got {actual}")
        return None
    return path


def _check_signature(precision: Precision, model: Path, where: str, problems: list[str]) -> None:
    interpreter = runtime.interpreter(model)
    for role, details, declared in (
        ("inputs", interpreter.get_input_details(), precision.inputs),
        ("outputs", interpreter.get_output_details(), precision.outputs),
    ):
        actual = tuple(runtime.describe(d) for d in details)
        if len(actual) != len(declared):
            problems.append(f"{where}.{role}: the model has {len(actual)}, the record declares {len(declared)}")
            continue
        for index, (have, want) in enumerate(zip(actual, declared, strict=True)):
            if have != want:
                problems.append(f"{where}.{role}[{index}]: record {want} does not match the model {have}")


def _check_state_pairs(record: Record, precision: Precision, where: str, problems: list[str]) -> None:
    for k, (i, o) in enumerate(precision.pair_indices(record.io)):
        state_in, state_out = precision.inputs[i], precision.outputs[o]
        label = f"{where}: state pair {k} ({state_in.name} <- {state_out.name})"
        if math.prod(state_in.shape) != math.prod(state_out.shape):
            problems.append(f"{label}: element counts differ")
        if state_in.dtype != state_out.dtype or state_in.zero_point != state_out.zero_point:
            problems.append(f"{label}: dtype or zero point differs")


def _check_golden(
    record: Record,
    precision: Precision,
    golden_path: Path,
    model: Path | None,
    replay: bool,
    where: str,
    problems: list[str],
) -> None:
    with np.load(golden_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    found = len(problems)
    golden.check_arrays(arrays, precision, precision.golden, record.io, f"{where}.golden", problems)
    if replay and model is not None and len(problems) == found:
        golden.replay_arrays(model, arrays, precision.golden, record.io, f"{where}.golden", problems)


def _check_v1(manifest: Manifest, v1_path: Path, problems: list[str]) -> None:
    """Every v1 entry must name the same files, hashes and runtime as the record that holds its model."""
    v1 = json.loads(Path(v1_path).read_text(encoding="utf-8"))
    by_model = {
        precision.model.path: (record, precision)
        for record in manifest.records
        for precision in record.precisions.values()
    }
    for item in v1.get("entries", []):
        where = f"v1 {item.get('id')}"
        if item.get("model") not in by_model:
            problems.append(f"{where}: no record lists the model {item.get('model')!r}")
            continue
        record, precision = by_model[item["model"]]
        golden = precision.golden
        pairs = [
            ("model_sha256", item["model_sha256"], precision.model.sha256),
            ("golden", item["golden"], golden and golden.file.path),
            ("golden_sha256", item["golden_sha256"], golden and golden.file.sha256),
            ("reference_runtime", item["reference_runtime"], golden and golden.reference_runtime),
            (
                "reference_runtime_version",
                item["reference_runtime_version"],
                golden and golden.reference_runtime_version,
            ),
            ("provenance_reference", item["provenance_reference"], record.card_file.path),
            ("license_reference", item["license_reference"], record.card_file.path),
        ]
        for key, old, new in pairs:
            if old != new:
                problems.append(f"{where}: {key} is {old!r} in v1 but {new!r} in record {record.id}")


def validate(
    root: Path,
    manifest: Manifest | None = None,
    *,
    signatures: bool = True,
    replay: bool = False,
    v1: Path | None = None,
    public: bool = True,
    cache: Path | None = None,
) -> None:
    """Validate every record against a hydrated checkout.

    Args:
        root: The repository checkout that ``lfs://`` and ``repo://`` paths resolve against.
        manifest: The records to check; by default the ones shipped with this package.
        signatures: Compare declared tensors and goldens with each model (needs the ``litert`` extra).
        replay: Also run every golden call (each batch row and sequence step) and require its outputs
            exactly, under the golden's recorded resolver and LiteRT version (a different installed
            version is reported, not compared).
        v1: A v1 corpus manifest whose entries must agree with the records holding their models.
        public: Refuse any record that is not public (the rule for this public repository), and fetch
            files from outside this repository without credentials, into a fresh cache unless ``cache``
            is given.
        cache: Where to fetch files from outside this repository (default: a fresh directory when
            ``public``, else ``hydrate.cache_dir()``).

    Raises:
        ValidationError: Listing every problem found.
    """
    root = Path(root).resolve()
    manifest = load_manifest() if manifest is None else manifest
    if replay and not signatures:
        raise ValueError("replay needs signature checks")
    problems: list[str] = []
    # A public check downloads into a fresh cache, so a cached copy cannot hide a file that needs credentials.
    fresh = tempfile.TemporaryDirectory(prefix="helia-zoo-public-") if public and cache is None else None
    if fresh is not None:
        cache = Path(fresh.name)
    try:
        _check_records(root, manifest, signatures, replay, public, cache, problems)
    finally:
        if fresh is not None:
            fresh.cleanup()
    if v1 is not None:
        _check_v1(manifest, v1, problems)
    if problems:
        raise ValidationError(problems)


def _check_records(
    root: Path,
    manifest: Manifest,
    signatures: bool,
    replay: bool,
    public: bool,
    cache: Path | None,
    problems: list[str],
) -> None:
    for record in manifest.records:
        where = f"record {record.id}"
        if public and record.visibility != "public":
            problems.append(f"{where}: visibility is {record.visibility!r}; this repository holds public records only")
            continue
        _check_file(root, record.card_file, f"{where}.card", problems, cache, public)
        for precision in record.precisions.values():
            pwhere = f"{where}.precisions.{precision.name}"
            _check_state_pairs(record, precision, pwhere, problems)
            model = _check_file(root, precision.model, f"{pwhere}.model", problems, cache, public)
            if model is not None and signatures:
                before = len(problems)
                _check_signature(precision, model, pwhere, problems)
                if len(problems) > before:
                    model = None
            if precision.golden is None:
                continue
            golden_file = _check_file(root, precision.golden.file, f"{pwhere}.golden", problems, cache, public)
            if golden_file is not None:
                _check_golden(record, precision, golden_file, model if signatures else None, replay, pwhere, problems)
