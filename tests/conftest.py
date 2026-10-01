# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import copy
import json
import os
from importlib import resources
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POINTER = b"version https://git-lfs.github.com/spec/v1"
PACKAGED = json.loads(resources.files("helia_model_zoo").joinpath("manifest.json").read_text(encoding="utf-8"))


@pytest.fixture
def data():
    """A mutable copy of the packaged manifest document."""
    return copy.deepcopy(PACKAGED)


@pytest.fixture(scope="session")
def root():
    """A checkout with Git LFS hydrated: HELIA_ZOO_ROOT, else this repository.

    Tests that need artifacts skip without one, unless HELIA_ZOO_REQUIRE_ARTIFACTS=1 (as in CI).
    """
    required = os.environ.get("HELIA_ZOO_REQUIRE_ARTIFACTS") == "1"
    path = Path(os.environ.get("HELIA_ZOO_ROOT", REPO)).resolve()
    model = path / "audio/rnnoise/model.tflite"
    if not model.is_file() or model.read_bytes()[: len(POINTER)] == POINTER:
        (pytest.fail if required else pytest.skip)(f"no hydrated checkout at {path}")
    if required:
        import ai_edge_litert  # noqa: F401
    else:
        pytest.importorskip("ai_edge_litert")
    return path


def entry(data, model_id):
    return next(e for e in data["entries"] if e["id"] == model_id)
