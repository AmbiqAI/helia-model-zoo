#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Generate deterministic LiteRT inputs and outputs without helia-aot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runs from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helia_model_zoo.cli import main as zoo  # noqa: E402


def main() -> int:
    """Write a single golden for a model, with inputs drawn from a seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolver", default="builtin_ref", choices=("builtin_ref", "builtin"))
    args = parser.parse_args()
    command = ["golden", "generate", str(args.model), str(args.output), "--seed", str(args.seed)]
    return zoo([*command, "--resolver", args.resolver])


if __name__ == "__main__":
    raise SystemExit(main())
