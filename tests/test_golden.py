# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest
from conftest import REPO, entry

import helia_model_zoo as zoo
from helia_model_zoo import golden
from helia_model_zoo.cli import main
from helia_model_zoo.golden import GoldenError, check, generate, reset_value, state_pairs_from_names
from helia_model_zoo.manifest import IO, FileRef, Golden, StatePair, Tensor, parse_manifest
from helia_model_zoo.validate import ValidationError, validate

DFNET2 = "audio/dfnet2/model.tflite"
RNNOISE = "audio/rnnoise/model.tflite"


@pytest.fixture
def dfnet2(root):
    item = zoo.get("dfnet2")
    return root / DFNET2, item.precision(), item.io


@pytest.fixture
def rnnoise(root):
    item = zoo.get("rnnoise")
    return root / RNNOISE, item.precision(), item.io


def meta(kind, steps, resets=()):
    return Golden(FileRef("repo://g.npz"), kind, steps, tuple(resets), None, "ai-edge-litert", "2.1.2", "builtin_ref")


def problems(arrays, precision, io, kind, steps, resets=()):
    found = []
    golden.check_arrays(arrays, precision, meta(kind, steps, resets), io, "g.golden", found)
    return found


def test_sequence_carries_state_and_resets(dfnet2):
    model, precision, io = dfnet2
    arrays = generate(model, precision, io, kind="sequence", steps=5, resets=[3], seed=1)
    state = arrays["input_2"]
    assert state.shape == (5, 1, 26304) and arrays["output_0"].shape == (5, 1, 1)
    assert np.array_equal(state[0], reset_value(precision.inputs[2]))
    assert np.array_equal(state[1], arrays["output_4"][0]) and np.array_equal(state[4], arrays["output_4"][3])
    assert np.array_equal(state[3], reset_value(precision.inputs[2])) and not np.array_equal(state[2], state[0])
    assert problems(arrays, precision, io, "sequence", 5, [3]) == []
    found = []
    golden.replay_arrays(model, arrays, meta("sequence", 5, [3]), io, "g.golden", found)
    assert found == []


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (2, "input_2 at step 2 is not output_4 of step 1"),
        (3, "input_2 at step 3 is not the reset value"),
        (0, "input_2 at step 0 is not the reset value"),
    ],
)
def test_sequence_carry_and_resets_are_checked(dfnet2, step, message):
    model, precision, io = dfnet2
    arrays = generate(model, precision, io, kind="sequence", steps=5, resets=[3], seed=1)
    arrays["input_2"] = arrays["input_2"].copy()
    arrays["input_2"][step].flat[0] += 1
    assert problems(arrays, precision, io, "sequence", 5, [3]) == [f"g.golden: {message}"]


def test_replay_checks_every_step(dfnet2):
    model, precision, io = dfnet2
    arrays = generate(model, precision, io, kind="sequence", steps=4, seed=1)
    arrays["output_1"] = arrays["output_1"].copy()
    arrays["output_1"][3].flat[0] += 1
    found = []
    golden.replay_arrays(model, arrays, meta("sequence", 4), io, "g.golden", found)
    assert found == ["g.golden: output_1 at step 3 does not replay exactly under builtin_ref (max abs difference 1)"]


def test_untied_quantized_state_cannot_be_a_sequence(rnnoise):
    model, precision, io = rnnoise
    with pytest.raises(GoldenError, match="state pair 0 differs in dtype, scale or zero point"):
        generate(model, precision, io, kind="sequence", steps=3)
    arrays = generate(model, precision, io, kind="batch", steps=3)
    assert problems(arrays, precision, io, "sequence", 3)[0] == (
        "g.golden: state pair 0 is not tied, so a sequence cannot carry it exactly"
    )


def test_batch_rows_start_from_reset(rnnoise):
    model, precision, io = rnnoise
    arrays = generate(model, precision, io, kind="batch", steps=3, seed=5)
    for pair in io.state_pairs:
        assert all(
            np.array_equal(row, reset_value(precision.inputs[pair.input])) for row in arrays[f"input_{pair.input}"]
        )
    assert problems(arrays, precision, io, "batch", 3) == []
    arrays["input_3"] = arrays["input_3"].copy()
    arrays["input_3"][1].flat[0] += 1
    assert problems(arrays, precision, io, "batch", 3) == ["g.golden: input_3 at step 1 is not the reset value"]


def test_reset_value_is_the_zero_point(rnnoise):
    _, precision, _ = rnnoise
    assert reset_value(precision.inputs[2]).flat[0] == -128 and reset_value(precision.inputs[1]).flat[0] == -1
    assert reset_value(Tensor("f", (2,), "float32", None, None)).tolist() == [0.0, 0.0]


def test_given_data_is_used_and_checked(dfnet2):
    model, precision, io = dfnet2
    erb = np.arange(3 * 32, dtype=np.int16).reshape(3, 1, 1, 1, 32)
    arrays = generate(model, precision, io, kind="sequence", steps=3, data={0: erb})
    assert np.array_equal(arrays["input_0"], erb)
    for bad, message in [
        ({0: erb[:2]}, "expected"),
        ({0: erb.astype(np.int32)}, "expected"),
        ({2: arrays["input_2"]}, "not a data input"),
        ({9: erb}, "not a data input"),
    ]:
        with pytest.raises(GoldenError, match=message):
            generate(model, precision, io, kind="sequence", steps=3, data=bad)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"kind": "single", "steps": 2}, "one step"),
        ({"kind": "batch", "steps": 3, "resets": [1]}, "only a sequence"),
        ({"kind": "sequence", "steps": 3, "resets": [3]}, "distinct steps"),
        ({"kind": "sequence", "steps": 3, "resets": [1, 1]}, "distinct steps"),
        ({"kind": "stream", "steps": 3}, "kind must be"),
    ],
)
def test_bad_requests_are_refused(dfnet2, options, message):
    model, precision, io = dfnet2
    with pytest.raises(GoldenError, match=message):
        generate(model, precision, io, **options)


def test_internal_state_sequences_are_refused(dfnet2):
    model, precision, _ = dfnet2
    with pytest.raises(GoldenError, match="internal_state"):
        generate(model, precision, IO("internal_state", ()), kind="sequence", steps=2)


def test_declared_tensors_must_match_the_model(dfnet2, rnnoise):
    model, _, io = dfnet2
    with pytest.raises(GoldenError, match="do not match"):
        generate(model, rnnoise[1], io)


def test_state_pairs_from_names():
    tensor = lambda name: Tensor(name, (1, 4), "int8", 0.5, 0)  # noqa: E731
    inputs = [tensor("serving_default_x:0"), tensor("serving_default_state_in_0:0"), tensor("state_in_1")]
    outputs = [tensor("StatefulPartitionedCall:0"), tensor("StatefulPartitionedCall:1"), tensor("y")]
    # LiteRT names outputs StatefulPartitionedCall:N, so tensor names alone leave the inputs unpaired.
    with pytest.raises(GoldenError, match="unpaired"):
        state_pairs_from_names(inputs, outputs)
    names = (["x", "state_in_0", "state_in_1"], ["state_out_1", "y", "state_out_0"])
    assert state_pairs_from_names(inputs, outputs, names) == (
        StatePair(1, 2, "zeros", True),
        StatePair(2, 0, "zeros", True),
    )
    with pytest.raises(GoldenError, match="unpaired"):
        state_pairs_from_names(inputs, outputs, (names[0], ["state_out_1", "y", "z"]))


def test_standalone_check(dfnet2, tmp_path):
    model, precision, io = dfnet2
    path = tmp_path / "g.npz"
    golden.write(path, generate(model, precision, io, kind="sequence", steps=4, resets=[2]))
    options = {"kind": "sequence", "steps": 4, "resets": [2], "state_pairs": [(2, 4)]}
    assert check(model, path, **options) == []
    assert check(model, path, **{**options, "resets": []}) == ["golden: input_2 at step 2 is not output_4 of step 1"]
    assert check(model, path, **{**options, "state_pairs": None}) == []  # no state names: arrays and replay only


def test_validate_a_sequence_entry(root, tmp_path, data, dfnet2):
    model, precision, io = dfnet2
    (tmp_path / "audio/dfnet2").mkdir(parents=True)
    for name in ("model.tflite", "README.md"):
        (tmp_path / "audio/dfnet2" / name).write_bytes((root / "audio/dfnet2" / name).read_bytes())
    path = tmp_path / "audio/dfnet2/sequence.npz"
    golden.write(path, generate(model, precision, io, kind="sequence", steps=4, resets=[2]))
    item = entry(data, "dfnet2")
    item["precisions"]["int16x8"]["golden"] = {
        "file": {
            "uri": "lfs://audio/dfnet2/sequence.npz",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        },
        "kind": "sequence",
        "steps": 4,
        "resets": [2],
        "source": {"seed": 42},
        "reference_runtime": "ai-edge-litert",
        "reference_runtime_version": "2.1.2",
        "resolver": "builtin_ref",
    }
    only = parse_manifest({"schema": data["schema"], "entries": [item]})
    validate(tmp_path, only, replay=True)
    item["precisions"]["int16x8"]["golden"]["resets"] = []
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, parse_manifest({"schema": data["schema"], "entries": [item]}))
    assert caught.value.problems == [
        "entry dfnet2.precisions.int16x8.golden: input_2 at step 2 is not output_4 of step 1"
    ]


def test_cli_generate_and_check(root, tmp_path, capsys):
    model = str(root / DFNET2)
    out = tmp_path / "seq.npz"
    args = [
        "golden",
        "generate",
        model,
        str(out),
        "--kind",
        "sequence",
        "--steps",
        "4",
        "--resets",
        "2",
        "--pair",
        "2:4",
    ]
    assert main([*args, "--print-manifest", "--uri", "lfs://audio/dfnet2/seq.npz"]) == 0
    block = json.loads(capsys.readouterr().out)
    assert block["file"] == {
        "uri": "lfs://audio/dfnet2/seq.npz",
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "bytes": out.stat().st_size,
    }
    assert (block["kind"], block["steps"], block["resets"], block["source"]) == ("sequence", 4, [2], {"seed": 42})
    check_args = ["golden", "check", model, str(out), "--kind", "sequence", "--steps", "4", "--pair", "2:4"]
    assert main([*check_args, "--resets", "2"]) == 0
    assert main(check_args) == 1
    assert "input_2 at step 2 is not output_4 of step 1" in capsys.readouterr().err
    entry_out = tmp_path / "entry.npz"
    assert (
        main(
            [
                "golden",
                "generate",
                model,
                str(entry_out),
                "--entry",
                "dfnet2-int16",
                "--kind",
                "sequence",
                "--steps",
                "3",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "golden",
                "generate",
                str(root / RNNOISE),
                str(tmp_path / "r.npz"),
                "--entry",
                "rnnoise",
                "--kind",
                "sequence",
                "--steps",
                "3",
            ]
        )
        == 1
    )
    assert "cannot be carried exactly" in capsys.readouterr().err


def test_generator_shim_keeps_its_command_line(root, tmp_path):
    out = tmp_path / "g.npz"
    subprocess.run(
        [sys.executable, str(REPO / "tools/generate_golden.py"), str(root / RNNOISE), str(out), "--seed", "3"],
        check=True,
        capture_output=True,
    )
    _, precision, _ = zoo.get("rnnoise"), zoo.get("rnnoise").precision(), None
    expected = generate(root / RNNOISE, precision, IO("stateless", ()), seed=3)
    with np.load(out) as written:
        assert sorted(written.files) == sorted(expected) and all(
            np.array_equal(written[k], expected[k]) for k in expected
        )


def test_tie_needs_the_zero_point_too(dfnet2):
    from dataclasses import replace

    model, precision, io = dfnet2
    arrays = generate(model, precision, io, kind="sequence", steps=3)
    out = precision.outputs[4]
    shifted = replace(precision, outputs=(*precision.outputs[:4], replace(out, zero_point=out.zero_point + 1)))
    assert problems(arrays, shifted, io, "sequence", 3) == [
        "g.golden: state pair 0 is not tied, so a sequence cannot carry it exactly"
    ]
    with pytest.raises(GoldenError, match="differs in dtype, scale or zero point"):
        generate(model, shifted, io, kind="sequence", steps=3)
