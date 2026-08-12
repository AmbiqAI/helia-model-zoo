#!/usr/bin/env python3
"""Generate deterministic LiteRT inputs and outputs without helia-aot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def _input_array(shape: tuple[int, ...], dtype: np.dtype, rng: np.random.Generator) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(info.min, -8)
        high = min(info.max, 7) + 1
        return rng.integers(low, high, size=shape, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(shape).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return rng.integers(0, 2, size=shape).astype(dtype)
    raise ValueError(f"unsupported input dtype: {dtype}")


def generate_golden(model_path: Path, output_path: Path, seed: int) -> None:
    """Run a model with deterministic generated inputs and save its I/O.

    Args:
        model_path: Hydrated TFLite model to execute.
        output_path: NPZ file to create.
        seed: NumPy random seed used for every generated input.
    """
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for index, detail in enumerate(interpreter.get_input_details()):
        value = _input_array(tuple(int(item) for item in detail["shape"]), np.dtype(detail["dtype"]), rng)
        arrays[f"input_{index}"] = value
        interpreter.set_tensor(detail["index"], value)
    interpreter.invoke()
    for index, detail in enumerate(interpreter.get_output_details()):
        arrays[f"output_{index}"] = interpreter.get_tensor(detail["index"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)


def main() -> int:
    """Parse CLI arguments and generate one golden fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_golden(args.model, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
