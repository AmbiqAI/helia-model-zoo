# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import copy
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POINTER = b"version https://git-lfs.github.com/spec/v1"
PACKAGED = {
    path.parent.name: json.loads(path.read_text(encoding="utf-8")) for path in REPO.glob("models/*/record.json")
}


@pytest.fixture
def data():
    """Mutable copies of the repository's record documents, keyed by model ID."""
    return copy.deepcopy(PACKAGED)


@pytest.fixture(scope="session")
def root():
    """A checkout with Git LFS hydrated: HELIA_ZOO_ROOT, else this repository.

    Tests that need artifacts skip without one, unless HELIA_ZOO_REQUIRE_ARTIFACTS=1 (as in CI).
    """
    required = os.environ.get("HELIA_ZOO_REQUIRE_ARTIFACTS") == "1"
    path = Path(os.environ.get("HELIA_ZOO_ROOT", REPO)).resolve()
    model = path / "models/rnnoise/a8w8/model.tflite"
    if not model.is_file() or model.read_bytes()[: len(POINTER)] == POINTER:
        (pytest.fail if required else pytest.skip)(f"no hydrated checkout at {path}")
    if required:
        import ai_edge_litert  # noqa: F401
    else:
        pytest.importorskip("ai_edge_litert")
    return path


def entry(data, model_id):
    return data[model_id]
