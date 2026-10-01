#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
"""Validate the model-zoo corpus manifest and its hydrated artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runs from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helia_model_zoo.v1 import validate_manifest  # noqa: E402


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
