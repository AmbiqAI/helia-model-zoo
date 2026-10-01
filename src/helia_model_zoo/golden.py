# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Golden fixtures (golden@2): generate, check and replay them.

A golden is an NPZ of ``input_i``/``output_i`` arrays in model I/O (subgraph) order, holding the
raw values fed to and produced by the model. ``single`` holds one call; ``batch`` and ``sequence``
add a leading axis of ``steps`` calls. In a ``batch`` every call starts from the reset state; in a
``sequence`` each explicit state input takes its paired output from the previous call, except at
step 0 and at the steps listed in ``resets``, where it takes the reset value. The reset value
``zeros`` is real-valued zero, which a quantized tensor stores as its zero point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import IO, FileRef, Golden, Precision, StatePair, Tensor

GOLDEN_KINDS = ("single", "batch", "sequence")


class GoldenError(ValueError):
    """A golden cannot be generated as asked."""


@dataclass(frozen=True)
class GoldenData:
    """A golden's arrays; ``batch`` and ``sequence`` goldens carry a leading step axis."""

    meta: Golden
    inputs: tuple[np.ndarray, ...]
    outputs: tuple[np.ndarray, ...]


def load_golden(path: Path, precision: Precision) -> GoldenData:
    """Load a golden NPZ for ``precision``.

    Raises:
        ValueError: If the archive's keys are not exactly the precision's inputs and outputs.
    """
    if precision.golden is None:
        raise ValueError(f"precision {precision.name} has no golden")
    inputs = [f"input_{i}" for i in range(len(precision.inputs))]
    outputs = [f"output_{i}" for i in range(len(precision.outputs))]
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {*inputs, *outputs}:
            raise ValueError(f"{path}: expected keys {sorted([*inputs, *outputs])}, found {sorted(arrays.files)}")
        return GoldenData(precision.golden, tuple(arrays[k] for k in inputs), tuple(arrays[k] for k in outputs))


def reset_value(tensor: Tensor) -> np.ndarray:
    """Real-valued zero for ``tensor``: its zero point when quantized."""
    return np.full(tensor.shape, tensor.zero_point or 0, dtype=tensor.dtype)


def _tied(precision: Precision, pair: StatePair) -> bool:
    state_in, state_out = precision.inputs[pair.input], precision.outputs[pair.output]
    return (state_in.scale, state_in.zero_point, state_in.dtype) == (
        state_out.scale,
        state_out.zero_point,
        state_out.dtype,
    )


def _random(tensor: Tensor, rng: np.random.Generator) -> np.ndarray:
    dtype = np.dtype(tensor.dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -8), min(info.max, 7) + 1, size=tensor.shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(tensor.shape).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return rng.integers(0, 2, size=tensor.shape).astype(dtype)
    raise GoldenError(f"unsupported input dtype: {dtype}")


def _check_request(precision: Precision, io: IO, kind: str, steps: int, resets: Sequence[int]) -> None:
    if kind not in GOLDEN_KINDS:
        raise GoldenError(f"kind must be one of {list(GOLDEN_KINDS)}, got {kind!r}")
    if steps < 1 or (kind == "single" and steps != 1):
        raise GoldenError("a single golden has one step; batch and sequence goldens need steps >= 1")
    if resets and kind != "sequence":
        raise GoldenError("only a sequence golden has resets")
    if any(not 0 < t < steps for t in resets) or len(set(resets)) != len(resets):
        raise GoldenError(f"resets must be distinct steps in [1, {steps}), got {list(resets)}")
    if kind == "sequence" and io.streaming == "internal_state":
        raise GoldenError("sequence goldens for internal_state models are not supported yet")
    if kind == "sequence":
        for k, pair in enumerate(io.state_pairs):
            if not _tied(precision, pair):
                raise GoldenError(
                    f"state pair {k} differs in dtype, scale or zero point, so its state cannot be carried exactly"
                )


def generate(
    model: Path,
    precision: Precision,
    io: IO,
    *,
    kind: str = "single",
    steps: int = 1,
    resets: Sequence[int] = (),
    data: Mapping[int, np.ndarray] | None = None,
    seed: int = 42,
    resolver: str = "builtin_ref",
) -> dict[str, np.ndarray]:
    """Run a model and return its golden arrays.

    Args:
        model: The TFLite model; ``precision`` must describe its inputs and outputs.
        precision: The declared tensors (its ``golden`` field is not used).
        io: The streaming kind and state pairs.
        kind: ``single``, ``batch`` or ``sequence``.
        steps: Calls in a batch or sequence.
        resets: Sequence steps whose state inputs return to the reset value.
        data: Values for non-state inputs by input index: ``[steps, *shape]`` (or ``shape`` for a
            single golden). Inputs not given are drawn from ``seed``.
        seed: Seed for drawn inputs (integers in [-8, 7], floats standard normal).
        resolver: LiteRT resolver, ``builtin_ref`` (reference kernels) by default.

    Raises:
        GoldenError: If the request cannot be met, for example an untied quantized state pair in
            a sequence.
    """
    from .runtime import describe, interpreter

    _check_request(precision, io, kind, steps, resets)
    loaded = interpreter(model, resolver)
    details_in, details_out = loaded.get_input_details(), loaded.get_output_details()
    actual = (tuple(describe(d) for d in details_in), tuple(describe(d) for d in details_out))
    if actual != (precision.inputs, precision.outputs):
        raise GoldenError("the declared tensors do not match the model")
    state = {pair.input: pair for pair in io.state_pairs}
    rng = np.random.default_rng(seed)
    given = {} if data is None else dict(data)
    for index, values in given.items():
        if index in state or not 0 <= index < len(precision.inputs):
            raise GoldenError(f"data for input {index}: not a data input of this model")
        tensor = precision.inputs[index]
        want = tensor.shape if kind == "single" else (steps, *tensor.shape)
        if values.shape != want or values.dtype.name != tensor.dtype:
            raise GoldenError(
                f"data for input {index}: expected {want} {tensor.dtype}, got {values.shape} {values.dtype}"
            )
    record_in: list[list[np.ndarray]] = [[] for _ in details_in]
    record_out: list[list[np.ndarray]] = [[] for _ in details_out]
    previous: list[np.ndarray] | None = None
    for t in range(steps):
        for index, (detail, tensor) in enumerate(zip(details_in, precision.inputs, strict=True)):
            if index in state:
                carry = kind == "sequence" and previous is not None and t not in resets
                value = previous[state[index].output].reshape(tensor.shape) if carry else reset_value(tensor)
            elif index in given:
                value = given[index] if kind == "single" else given[index][t]
            else:
                value = _random(tensor, rng)
            loaded.set_tensor(detail["index"], value)
            record_in[index].append(np.array(value))
        loaded.invoke()
        previous = [np.array(loaded.get_tensor(d["index"])) for d in details_out]
        for index, value in enumerate(previous):
            record_out[index].append(value)
    stack = (lambda xs: xs[0]) if kind == "single" else np.stack
    arrays = {f"input_{i}": stack(xs) for i, xs in enumerate(record_in)}
    arrays |= {f"output_{i}": stack(xs) for i, xs in enumerate(record_out)}
    return arrays


def write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write golden arrays as an uncompressed NPZ."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def check_arrays(
    arrays: Mapping[str, np.ndarray], precision: Precision, golden: Golden, io: IO, where: str, problems: list[str]
) -> None:
    """Check keys, shapes and dtypes, and for batch and sequence goldens the state inputs.

    Each problem is appended to ``problems`` prefixed with ``where``.
    """
    expected = {f"input_{i}": t for i, t in enumerate(precision.inputs)}
    expected |= {f"output_{i}": t for i, t in enumerate(precision.outputs)}
    if set(arrays) != expected.keys():
        problems.append(f"{where}: expected keys {sorted(expected)}, found {sorted(arrays)}")
        return
    found = len(problems)
    lead = () if golden.kind == "single" else (golden.steps,)
    for key, tensor in expected.items():
        array = arrays[key]
        shape = lead + tensor.shape
        if len(array.shape) != len(shape) or any(w != -1 and a != w for a, w in zip(array.shape, shape)):
            problems.append(f"{where}: {key} shape {array.shape}, expected {shape}")
        if array.dtype.name != tensor.dtype:
            problems.append(f"{where}: {key} dtype {array.dtype.name}, expected {tensor.dtype}")
    if len(problems) > found or golden.kind == "single":
        return
    for k, pair in enumerate(io.state_pairs):
        state_in = precision.inputs[pair.input]
        if golden.kind == "sequence" and not _tied(precision, pair):
            problems.append(f"{where}: state pair {k} is not tied, so a sequence cannot carry it exactly")
            continue
        fed, produced = arrays[f"input_{pair.input}"], arrays[f"output_{pair.output}"]
        for t in range(golden.steps):
            starts = golden.kind == "batch" or t == 0 or t in golden.resets
            want = reset_value(state_in) if starts else produced[t - 1].reshape(state_in.shape)
            if not np.array_equal(fed[t], want):
                what = "the reset value" if starts else f"output_{pair.output} of step {t - 1}"
                problems.append(f"{where}: input_{pair.input} at step {t} is not {what}")
                break


def replay_arrays(
    model: Path, arrays: Mapping[str, np.ndarray], golden: Golden, io: IO, where: str, problems: list[str]
) -> None:
    """Run every stored step under the golden's resolver and require each output exactly."""
    from .runtime import interpreter, runtime_version

    installed = runtime_version(golden.reference_runtime)
    if installed != golden.reference_runtime_version:
        problems.append(
            f"{where}: replay needs {golden.reference_runtime} {golden.reference_runtime_version}; "
            f"installed: {installed}"
        )
        return
    if golden.kind != "single" and io.streaming == "internal_state":
        problems.append(f"{where}: replay of {golden.kind} goldens for internal_state models is not supported")
        return
    loaded = interpreter(model, golden.resolver)
    details_in, details_out = loaded.get_input_details(), loaded.get_output_details()
    steps = [None] if golden.kind == "single" else range(golden.steps)
    for t in steps:
        pick = (lambda a: a) if t is None else (lambda a: a[t])
        try:
            for index, detail in enumerate(details_in):
                loaded.set_tensor(detail["index"], pick(arrays[f"input_{index}"]))
            loaded.invoke()
        except ValueError as error:
            problems.append(f"{where}: replay failed: {error}")
            return
        for index, detail in enumerate(details_out):
            actual, stored = loaded.get_tensor(detail["index"]), pick(arrays[f"output_{index}"])
            if not np.array_equal(actual, stored):
                worst = np.max(np.abs(actual.astype(np.float64) - stored.astype(np.float64)))
                at = "" if t is None else f" at step {t}"
                problems.append(
                    f"{where}: output_{index}{at} does not replay exactly under {golden.resolver} "
                    f"(max abs difference {worst:g})"
                )
                return


def state_pairs_from_names(
    inputs: Sequence[Tensor], outputs: Sequence[Tensor], names: tuple[Sequence[str], Sequence[str]] | None = None
) -> tuple[StatePair, ...]:
    """Pairs named ``state_in_k``/``state_out_k`` (helia-edge's convention); empty if none.

    ``names`` gives each input's and output's signature name, which LiteRT keeps when the tensor
    names differ (outputs are often ``StatefulPartitionedCall:N``); tensor names are used otherwise.

    Raises:
        GoldenError: If a ``state_in_k`` has no ``state_out_k`` or the reverse.
    """

    def numbered(labels: Sequence[str], stem: str) -> dict[int, int]:
        found = {}
        for index, label in enumerate(labels):
            match = re.search(rf"(?:^|[^a-z0-9]){stem}_(\d+)(?:$|[^0-9])", label)
            if match:
                found[int(match[1])] = index
        return found

    in_names, out_names = names or ([t.name for t in inputs], [t.name for t in outputs])
    ins, outs = numbered(in_names, "state_in"), numbered(out_names, "state_out")
    if ins.keys() != outs.keys():
        raise GoldenError(f"unpaired state tensors: inputs {sorted(ins)}, outputs {sorted(outs)}")
    return tuple(
        StatePair(ins[k], outs[k], "zeros", inputs[ins[k]].scale == outputs[outs[k]].scale) for k in sorted(ins)
    )


def check(
    model: Path,
    golden_path: Path,
    *,
    kind: str = "single",
    steps: int = 1,
    resets: Sequence[int] = (),
    state_pairs: Sequence[tuple[int, int]] | None = None,
    resolver: str = "builtin_ref",
    replay: bool = True,
    reference_runtime_version: str | None = None,
) -> list[str]:
    """Check one golden file against its model, without a manifest entry.

    ``state_pairs`` are ``(input index, output index)``; by default they come from
    ``state_in_k``/``state_out_k`` tensor names. Returns the problems found (empty when it passes).
    """
    from .runtime import model_tensors, runtime_version, signature_names

    inputs, outputs = model_tensors(model)
    if state_pairs is None:
        pairs = state_pairs_from_names(inputs, outputs, signature_names(model))
    else:
        pairs = tuple(StatePair(i, o, "zeros", inputs[i].scale == outputs[o].scale) for i, o in state_pairs)
    io = IO("explicit_state" if pairs else "stateless", pairs)
    version = reference_runtime_version or runtime_version("ai-edge-litert") or "unknown"
    file = FileRef(f"repo://{Path(golden_path).name}")
    meta = Golden(file, kind, steps, tuple(resets), None, "ai-edge-litert", version, resolver)
    precision = Precision("check", FileRef(f"repo://{Path(model).name}"), inputs, outputs, meta)
    problems: list[str] = []
    with np.load(golden_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    check_arrays(arrays, precision, meta, io, "golden", problems)
    if replay and not problems:
        replay_arrays(model, arrays, meta, io, "golden", problems)
    return problems


def manifest_block(path: Path, golden: Golden) -> dict[str, Any]:
    """The manifest ``golden`` object for a written golden file at ``path``."""
    from .hydrate import sha256_file

    return {
        "file": {"uri": golden.file.uri, "sha256": sha256_file(path), "bytes": Path(path).stat().st_size},
        "kind": golden.kind,
        "steps": golden.steps,
        "resets": list(golden.resets),
        "source": golden.source,
        "reference_runtime": golden.reference_runtime,
        "reference_runtime_version": golden.reference_runtime_version,
        "resolver": golden.resolver,
    }


__all__ = [
    "GOLDEN_KINDS",
    "GoldenData",
    "GoldenError",
    "check",
    "check_arrays",
    "generate",
    "load_golden",
    "manifest_block",
    "replay_arrays",
    "reset_value",
    "state_pairs_from_names",
    "write",
]
