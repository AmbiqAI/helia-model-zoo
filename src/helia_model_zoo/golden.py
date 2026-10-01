# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Golden fixtures: ``input_i``/``output_i`` arrays in model I/O order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .manifest import Golden, Precision


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
