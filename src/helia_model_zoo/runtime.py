# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""LiteRT access for signature checks, golden generation and replay (the ``litert`` extra)."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Tensor


def litert_module() -> Any:
    """``ai_edge_litert.interpreter``.

    Raises:
        ImportError: Naming the ``litert`` extra when LiteRT is not installed.
    """
    try:
        from ai_edge_litert import interpreter
    except ImportError as error:
        raise ImportError("signature and replay checks need LiteRT: install helia-model-zoo[litert]") from error
    return interpreter


def runtime_version(distribution: str) -> str | None:
    """The installed version of a distribution, or None."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def interpreter(model: Path, resolver: str | None = None) -> Any:
    """An allocated LiteRT interpreter; ``resolver`` is ``builtin``, ``builtin_ref`` or None (LiteRT's default)."""
    litert = litert_module()
    kinds = {
        None: litert.OpResolverType.AUTO,
        "builtin": litert.OpResolverType.BUILTIN,
        "builtin_ref": litert.OpResolverType.BUILTIN_REF,
    }
    result = litert.Interpreter(model_path=str(model), experimental_op_resolver_type=kinds[resolver])
    result.allocate_tensors()
    return result


def describe(detail: dict[str, Any]) -> Tensor:
    """A LiteRT input or output detail as a manifest tensor."""
    scale, zero_point = detail["quantization"]
    quantized = scale != 0.0
    return Tensor(
        detail["name"],
        tuple(int(d) for d in detail["shape"]),
        np.dtype(detail["dtype"]).name,
        float(scale) if quantized else None,
        int(zero_point) if quantized else None,
    )


def model_tensors(model: Path) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """A model's inputs and outputs, in subgraph order."""
    loaded = interpreter(model)
    return (
        tuple(describe(d) for d in loaded.get_input_details()),
        tuple(describe(d) for d in loaded.get_output_details()),
    )


def signature_names(model: Path) -> tuple[list[str], list[str]] | None:
    """Each input's and output's signature name, in subgraph order; None without a single signature."""
    loaded = interpreter(model)
    signatures = loaded.get_signature_list()
    if len(signatures) != 1:
        return None
    runner = loaded.get_signature_runner(next(iter(signatures)))
    by_index = {d["index"]: name for name, d in runner.get_input_details().items()}
    by_index |= {d["index"]: name for name, d in runner.get_output_details().items()}
    names = (
        [by_index.get(d["index"], d["name"]) for d in loaded.get_input_details()],
        [by_index.get(d["index"], d["name"]) for d in loaded.get_output_details()],
    )
    return names
