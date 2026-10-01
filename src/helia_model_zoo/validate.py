# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Validate a manifest@2 against a hydrated checkout of this repository."""

from __future__ import annotations

import hashlib
import json
import math
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Entry, FileRef, Manifest, Precision, Tensor, load_manifest

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ValidationError(ValueError):
    """One or more problems were found; ``problems`` lists them all."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(root: Path, ref: FileRef, where: str, problems: list[str]) -> Path | None:
    """Check that a file exists and, for an artifact, that its bytes are hydrated and match."""
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
    if path.stat().st_size != ref.bytes:
        problems.append(f"{where}: {ref.uri} is {path.stat().st_size} bytes, expected {ref.bytes}")
        return None
    actual = sha256_file(path)
    if actual != ref.sha256:
        problems.append(f"{where}: sha256 mismatch for {ref.uri}: expected {ref.sha256}, got {actual}")
        return None
    return path


def _litert() -> Any:
    try:
        from ai_edge_litert import interpreter
    except ImportError as error:
        raise ImportError("signature and replay checks need LiteRT: install helia-model-zoo[litert]") from error
    return interpreter


def _runtime_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _interpreter(model: Path, resolver: str | None = None) -> Any:
    litert = _litert()
    kinds = {
        None: litert.OpResolverType.AUTO,
        "builtin": litert.OpResolverType.BUILTIN,
        "builtin_ref": litert.OpResolverType.BUILTIN_REF,
    }
    interpreter = litert.Interpreter(model_path=str(model), experimental_op_resolver_type=kinds[resolver])
    interpreter.allocate_tensors()
    return interpreter


def _describe(detail: dict[str, Any]) -> Tensor:
    scale, zero_point = detail["quantization"]
    quantized = scale != 0.0
    return Tensor(
        detail["name"],
        tuple(int(d) for d in detail["shape"]),
        np.dtype(detail["dtype"]).name,
        float(scale) if quantized else None,
        int(zero_point) if quantized else None,
    )


def _check_signature(precision: Precision, model: Path, where: str, problems: list[str]) -> None:
    interpreter = _interpreter(model)
    for role, details, declared in (
        ("inputs", interpreter.get_input_details(), precision.inputs),
        ("outputs", interpreter.get_output_details(), precision.outputs),
    ):
        actual = tuple(_describe(d) for d in details)
        if len(actual) != len(declared):
            problems.append(f"{where}.{role}: the model has {len(actual)}, the manifest declares {len(declared)}")
            continue
        for index, (have, want) in enumerate(zip(actual, declared, strict=True)):
            if have != want:
                problems.append(f"{where}.{role}[{index}]: manifest {want} does not match the model {have}")


def _check_state_pairs(entry: Entry, precision: Precision, where: str, problems: list[str]) -> None:
    for k, pair in enumerate(entry.io.state_pairs):
        state_in, state_out = precision.inputs[pair.input], precision.outputs[pair.output]
        label = f"{where}: state pair {k} ({state_in.name} <- {state_out.name})"
        if math.prod(state_in.shape) != math.prod(state_out.shape):
            problems.append(f"{label}: element counts differ")
        if state_in.dtype != state_out.dtype or state_in.zero_point != state_out.zero_point:
            problems.append(f"{label}: dtype or zero point differs")
        if pair.scales_tied != (state_in.scale == state_out.scale):
            problems.append(
                f"{label}: scales_tied is {pair.scales_tied}, but the scales are {state_in.scale} and {state_out.scale}"
            )


def _check_golden(
    precision: Precision, golden_path: Path, model: Path | None, replay: bool, where: str, problems: list[str]
) -> None:
    golden = precision.golden
    found = len(problems)
    expected = {f"input_{i}": t for i, t in enumerate(precision.inputs)}
    expected |= {f"output_{i}": t for i, t in enumerate(precision.outputs)}
    with np.load(golden_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected.keys():
            problems.append(f"{where}.golden: expected keys {sorted(expected)}, found {sorted(arrays.files)}")
            return
        lead = () if golden.kind == "single" else (golden.steps,)
        for key, tensor in expected.items():
            array = arrays[key]
            shape = lead + tensor.shape
            if len(array.shape) != len(shape) or any(w != -1 and a != w for a, w in zip(array.shape, shape)):
                problems.append(f"{where}.golden: {key} shape {array.shape}, expected {shape}")
            if array.dtype.name != tensor.dtype:
                problems.append(f"{where}.golden: {key} dtype {array.dtype.name}, expected {tensor.dtype}")
        if not replay or model is None or len(problems) > found:
            return
        if golden.kind != "single":
            problems.append(f"{where}.golden: replay of {golden.kind} goldens is not supported yet")
            return
        installed = _runtime_version(golden.reference_runtime)
        if installed != golden.reference_runtime_version:
            problems.append(
                f"{where}.golden: replay needs {golden.reference_runtime} "
                f"{golden.reference_runtime_version}; installed: {installed}"
            )
            return
        interpreter = _interpreter(model, golden.resolver)
        try:
            for index, detail in enumerate(interpreter.get_input_details()):
                interpreter.set_tensor(detail["index"], arrays[f"input_{index}"])
            interpreter.invoke()
        except ValueError as error:
            problems.append(f"{where}.golden: replay failed: {error}")
            return
        for index, detail in enumerate(interpreter.get_output_details()):
            actual, stored = interpreter.get_tensor(detail["index"]), arrays[f"output_{index}"]
            if not np.array_equal(actual, stored):
                worst = np.max(np.abs(actual.astype(np.float64) - stored.astype(np.float64)))
                problems.append(
                    f"{where}.golden: output_{index} does not replay exactly under "
                    f"{golden.resolver} (max abs difference {worst:g})"
                )


def _check_v1(manifest: Manifest, v1_path: Path, problems: list[str]) -> None:
    """Every v1 entry must equal the v2 precision its ID aliases."""
    v1 = json.loads(Path(v1_path).read_text(encoding="utf-8"))
    for item in v1.get("entries", []):
        where = f"v1 {item.get('id')}"
        try:
            entry, precision = manifest.resolve(item["id"])
        except KeyError:
            problems.append(f"{where}: no v2 entry has this alias")
            continue
        if precision is None:
            problems.append(f"{where}: is a v2 ID, not an alias naming a precision")
            continue
        golden = precision.golden
        pairs = [
            ("model", f"lfs://{item['model']}", precision.model.uri),
            ("model_sha256", item["model_sha256"], precision.model.sha256),
            ("golden", f"lfs://{item['golden']}", golden and golden.file.uri),
            ("golden_sha256", item["golden_sha256"], golden and golden.file.sha256),
            ("reference_runtime", item["reference_runtime"], golden and golden.reference_runtime),
            (
                "reference_runtime_version",
                item["reference_runtime_version"],
                golden and golden.reference_runtime_version,
            ),
            ("provenance_reference", f"repo://{item['provenance_reference']}", entry.card.uri),
            ("license_reference", f"repo://{item['license_reference']}", entry.license.reference.uri),
        ]
        for key, old, new in pairs:
            if old != new:
                problems.append(f"{where}: {key} is {old!r} in v1 but {new!r} in v2")


def validate(
    root: Path,
    manifest: Manifest | None = None,
    *,
    signatures: bool = True,
    replay: bool = False,
    v1: Path | None = None,
    public: bool = True,
) -> None:
    """Validate every entry of a manifest against a hydrated checkout.

    Args:
        root: The repository checkout that ``lfs://`` and ``repo://`` paths resolve against.
        manifest: The manifest to check; by default the one shipped with this package.
        signatures: Compare declared tensors and goldens with each model (needs the ``litert`` extra).
        replay: Also run each single golden and require its outputs exactly, under its recorded resolver
            and its recorded LiteRT version (a different installed version is reported, not compared).
        v1: A frozen v1 manifest whose entries must equal their v2 aliases.
        public: Refuse any entry that is not public (the rule for this public repository).

    Raises:
        ValidationError: Listing every problem found.
    """
    root = Path(root).resolve()
    manifest = load_manifest() if manifest is None else manifest
    if replay and not signatures:
        raise ValueError("replay needs signature checks")
    problems: list[str] = []
    for entry in manifest.entries:
        where = f"entry {entry.id}"
        if public and entry.visibility != "public":
            problems.append(f"{where}: visibility is {entry.visibility!r}; this manifest holds public entries only")
            continue
        _check_file(root, entry.card, f"{where}.card", problems)
        _check_file(root, entry.license.reference, f"{where}.license.reference", problems)
        for precision in entry.precisions.values():
            pwhere = f"{where}.precisions.{precision.name}"
            _check_state_pairs(entry, precision, pwhere, problems)
            model = _check_file(root, precision.model, f"{pwhere}.model", problems)
            if model is not None and signatures:
                before = len(problems)
                _check_signature(precision, model, pwhere, problems)
                if len(problems) > before:
                    model = None
            if precision.golden is None:
                continue
            golden = _check_file(root, precision.golden.file, f"{pwhere}.golden", problems)
            if golden is not None:
                _check_golden(precision, golden, model if signatures else None, replay, pwhere, problems)
    if v1 is not None:
        _check_v1(manifest, v1, problems)
    if problems:
        raise ValidationError(problems)
