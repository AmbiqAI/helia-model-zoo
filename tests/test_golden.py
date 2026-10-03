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
from helia_model_zoo.manifest import IO, FileRef, Golden, StatePair, Tensor, parse_records
from helia_model_zoo.validate import ValidationError, validate

DFNET2 = "models/dfnet2/a16w8/model.tflite"
RNNOISE = "models/rnnoise/a8w8/model.tflite"


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
    for i, _ in precision.pair_indices(io):
        assert all(np.array_equal(row, reset_value(precision.inputs[i])) for row in arrays[f"input_{i}"])
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
        ({"kind": "sequence", "steps": 3, "resets": [0]}, "distinct steps"),
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
        StatePair("serving_default_state_in_0:0", "y"),
        StatePair("state_in_1", "StatefulPartitionedCall:0"),
    )
    with pytest.raises(GoldenError, match="unpaired"):
        state_pairs_from_names(inputs, outputs, (names[0], ["state_out_1", "y", "z"]))


def test_a_pair_needs_uniquely_named_tensors():
    tensor = lambda name: Tensor(name, (1, 4), "int8", 0.5, 0)  # noqa: E731
    inputs, outputs = [tensor("x"), tensor("x")], [tensor("y")]
    with pytest.raises(GoldenError, match="'x' is not unique"):
        golden.pairs_by_index(inputs, outputs, [(1, 0)])
    with pytest.raises(GoldenError, match="'y' is not unique"):
        golden.pairs_by_index([tensor("a")], [tensor("y"), tensor("y")], [(0, 1)])
    # Signature names can pair a tensor whose name another tensor shares.
    with pytest.raises(GoldenError, match="'x' is not unique"):
        state_pairs_from_names(inputs, outputs, (["data", "state_in_0"], ["state_out_0"]))


def test_standalone_check(dfnet2, tmp_path):
    model, precision, io = dfnet2
    path = tmp_path / "g.npz"
    golden.write(path, generate(model, precision, io, kind="sequence", steps=4, resets=[2]))
    options = {"kind": "sequence", "steps": 4, "resets": [2], "state_pairs": [(2, 4)]}
    assert check(model, path, **options) == []
    assert check(model, path, **{**options, "resets": []}) == ["golden: input_2 at step 2 is not output_4 of step 1"]
    # DFNet2 has no state_in_k/state_out_k names, so without --pair a sequence check has nothing to carry.
    assert check(model, path, **{**options, "state_pairs": None}) == [
        "golden: a sequence golden needs explicit state pairs; use a batch for a stateless model"
    ]


def test_validate_a_sequence_entry(root, tmp_path, data, dfnet2):
    model, precision, io = dfnet2
    (tmp_path / "models/dfnet2/a16w8").mkdir(parents=True)
    for name in ("a16w8/model.tflite", "README.md"):
        (tmp_path / "models/dfnet2" / name).write_bytes((root / "models/dfnet2" / name).read_bytes())
    path = tmp_path / "models/dfnet2/a16w8/sequence.npz"
    golden.write(path, generate(model, precision, io, kind="sequence", steps=4, resets=[2]))
    item = entry(data, "dfnet2")
    item["precisions"]["a16w8"]["golden"] = {
        "path": "a16w8/sequence.npz",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "kind": "sequence",
        "steps": 4,
        "resets": [2],
        "source": {"seed": 42},
        "runtime": "ai-edge-litert==2.1.2",
        "resolver": "builtin_ref",
    }
    only = parse_records({"dfnet2": item})
    validate(tmp_path, only, replay=True)
    item["precisions"]["a16w8"]["golden"]["resets"] = []
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, parse_records({"dfnet2": item}))
    assert caught.value.problems == [
        "record dfnet2.precisions.a16w8.golden: input_2 at step 2 is not output_4 of step 1"
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
    assert main([*args, "--print-record", "--path", "a16w8/seq.npz"]) == 0
    block = json.loads(capsys.readouterr().out)
    assert (block["path"], block["sha256"], block["bytes"]) == (
        "a16w8/seq.npz",
        hashlib.sha256(out.read_bytes()).hexdigest(),
        out.stat().st_size,
    )
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
                "--record",
                "dfnet2",
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
                "--record",
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


def tamper(arrays, key, step):
    arrays[key] = arrays[key].copy()
    arrays[key][step].flat[0] += 1


def test_last_step_is_checked_and_replayed(dfnet2):
    model, precision, io = dfnet2
    arrays = generate(model, precision, io, kind="sequence", steps=5, seed=2)
    carried = dict(arrays)
    tamper(carried, "input_2", 4)
    assert problems(carried, precision, io, "sequence", 5) == ["g.golden: input_2 at step 4 is not output_4 of step 3"]
    replayed = dict(arrays)
    tamper(replayed, "output_0", 4)
    found = []
    golden.replay_arrays(model, replayed, meta("sequence", 5), io, "g.golden", found)
    assert found == ["g.golden: output_0 at step 4 does not replay exactly under builtin_ref (max abs difference 1)"]


def test_single_golden_state_starts_from_reset(dfnet2):
    model, precision, io = dfnet2
    arrays = generate(model, precision, io, seed=4)
    assert np.array_equal(arrays["input_2"], reset_value(precision.inputs[2]))


def test_internal_state_is_single_only(dfnet2):
    model, precision, _ = dfnet2
    internal = IO("internal_state", ())
    for kind in ("batch", "sequence"):
        with pytest.raises(GoldenError, match=f"{kind} goldens for internal_state models are not supported"):
            generate(model, precision, internal, kind=kind, steps=2)
    arrays = generate(model, precision, IO("stateless", ()), kind="batch", steps=2)
    assert problems(arrays, precision, internal, "batch", 2) == [
        "g.golden: batch goldens for internal_state models are not supported yet"
    ]


def test_sequence_without_pairs_is_refused(dfnet2):
    model, precision, _ = dfnet2
    with pytest.raises(GoldenError, match="needs explicit state pairs"):
        generate(model, precision, IO("stateless", ()), kind="sequence", steps=2)


def test_carry_reshapes_the_output(rnnoise):
    from dataclasses import replace

    _, precision, _ = rnnoise
    # RNNoise state inputs are [1, 24] and its outputs [1, 1, 24]; tie pair 0 for the check.
    out = precision.outputs[3]
    tied = replace(
        precision, outputs=(*precision.outputs[:3], replace(out, scale=precision.inputs[1].scale), precision.outputs[4])
    )
    io = IO("explicit_state", (StatePair(tied.inputs[1].name, tied.outputs[3].name),))
    produced = np.arange(3 * 24, dtype=np.int8).reshape(3, 1, 1, 24)
    fed = np.stack([reset_value(tied.inputs[1]), produced[0].reshape(1, 24), produced[1].reshape(1, 24)])
    arrays = {f"input_{i}": np.stack([reset_value(t)] * 3) for i, t in enumerate(tied.inputs)}
    arrays |= {f"output_{i}": np.stack([reset_value(t)] * 3) for i, t in enumerate(tied.outputs)}
    arrays["input_1"], arrays["output_3"] = fed, produced
    assert problems(arrays, tied, io, "sequence", 3) == []


def test_tie_needs_the_dtype_too(dfnet2):
    from dataclasses import replace

    model, precision, io = dfnet2
    out = precision.outputs[4]
    widened = replace(precision, outputs=(*precision.outputs[:4], replace(out, dtype="int32")))
    with pytest.raises(GoldenError, match="differs in dtype, scale or zero point"):
        generate(model, widened, io, kind="sequence", steps=2)


def test_golden_files_are_uncompressed(dfnet2, tmp_path):
    import zipfile

    model, precision, io = dfnet2
    golden.write(tmp_path / "g.npz", generate(model, precision, io))
    with zipfile.ZipFile(tmp_path / "g.npz") as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}


def test_standalone_check_replays_with_its_resolver_and_version(rnnoise, tmp_path):
    model, precision, io = rnnoise
    path = tmp_path / "b.npz"
    arrays = generate(model, precision, io, kind="batch", steps=3, seed=9)
    golden.write(path, arrays)
    options = {"kind": "batch", "steps": 3, "state_pairs": []}
    assert check(model, path, **options) == []
    assert any("does not replay exactly under builtin" in p for p in check(model, path, **options, resolver="builtin"))
    assert check(model, path, **options, reference_runtime_version="0.0.1")[0].startswith(
        "golden: replay needs ai-edge-litert 0.0.1"
    )
    tamper(arrays, "output_1", 2)
    golden.write(path, arrays)
    assert check(model, path, **options)[0].startswith("golden: output_1 at step 2 does not replay exactly")
    assert check(model, path, **options, replay=False) == []


def test_validate_replays_every_sequence_step(root, tmp_path, data, dfnet2):
    model, precision, io = dfnet2
    (tmp_path / "models/dfnet2/a16w8").mkdir(parents=True)
    for name in ("a16w8/model.tflite", "README.md"):
        (tmp_path / "models/dfnet2" / name).write_bytes((root / "models/dfnet2" / name).read_bytes())
    path = tmp_path / "models/dfnet2/a16w8/sequence.npz"
    arrays = generate(model, precision, io, kind="sequence", steps=3)
    tamper(arrays, "output_3", 2)
    golden.write(path, arrays)
    item = entry(data, "dfnet2")
    item["precisions"]["a16w8"]["golden"] = {
        "path": "a16w8/sequence.npz",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "kind": "sequence",
        "steps": 3,
        "resets": [],
        "source": {"seed": 42},
        "runtime": "ai-edge-litert==2.1.2",
        "resolver": "builtin_ref",
    }
    only = parse_records({"dfnet2": item})
    validate(tmp_path, only)
    with pytest.raises(ValidationError) as caught:
        validate(tmp_path, only, replay=True)
    assert caught.value.problems[0].startswith(
        "record dfnet2.precisions.a16w8.golden: output_3 at step 2 does not replay exactly"
    )


def test_cli_check_no_replay(root, tmp_path, dfnet2):
    model, precision, io = dfnet2
    path = tmp_path / "g.npz"
    arrays = generate(model, precision, io, kind="sequence", steps=3)
    tamper(arrays, "output_0", 1)
    golden.write(path, arrays)
    base = ["golden", "check", str(model), str(path), "--kind", "sequence", "--steps", "3", "--pair", "2:4"]
    assert main(base) == 1
    assert main([*base, "--no-replay"]) == 0


@pytest.mark.parametrize(
    ("extra", "code", "message"),
    [
        (["--pair", "2"], 1, "--pair expects IN:OUT"),
        (["--pair", "9:9"], 1, "out of range"),
        (["--record", "dfnet2", "--pair", "2:4"], 2, "cannot be combined"),
    ],
)
def test_cli_generate_argument_errors(root, tmp_path, capsys, extra, code, message):
    args = ["golden", "generate", str(root / DFNET2), str(tmp_path / "g.npz"), "--kind", "sequence", "--steps", "2"]
    assert main([*args, *extra]) == code
    assert message in capsys.readouterr().err


def test_cli_generate_data_errors(root, tmp_path, capsys):
    bad = tmp_path / "bad.npz"
    np.savez(bad, erb=np.zeros((2, 1, 1, 1, 32), np.int16))
    args = [
        "golden",
        "generate",
        str(root / DFNET2),
        str(tmp_path / "g.npz"),
        "--kind",
        "sequence",
        "--steps",
        "2",
        "--pair",
        "2:4",
    ]
    assert main([*args, "--data", str(bad)]) == 1
    assert "--data keys must be input_N" in capsys.readouterr().err
    partial = tmp_path / "partial.npz"
    np.savez(partial, input_0=np.zeros((2, 1, 1, 1, 32), np.int16))
    recorded = ["--print-record", "--source-uri", "https://example.com/a.wav", "--source-sha256", "0" * 64]
    assert main([*args, "--data", str(partial), *recorded]) == 1
    assert "missing [1]" in capsys.readouterr().err
    assert not (tmp_path / "g.npz").exists()
    assert main([*args, "--data", str(partial), "--print-record"]) == 2
    assert not (tmp_path / "g.npz").exists()


def test_cli_check_pair_errors(root, tmp_path, capsys, dfnet2):
    model, precision, io = dfnet2
    path = tmp_path / "g.npz"
    golden.write(path, generate(model, precision, io))
    assert main(["golden", "check", str(model), str(path), "--pair", "x"]) == 2
    assert main(["golden", "check", str(model), str(path), "--pair", "9:9"]) == 1
    assert "state pairs" in capsys.readouterr().err


def test_state_names_match_exactly():
    tensor = lambda name: Tensor(name, (1, 4), "int8", 0.5, 0)  # noqa: E731
    inputs, outputs = [tensor("a"), tensor("b")], [tensor("c"), tensor("d")]
    assert state_pairs_from_names(inputs, outputs, (["state_in_01", "my_state_in_1"], ["c", "d"])) == ()
    with pytest.raises(GoldenError, match="two tensors are named state_in_0"):
        state_pairs_from_names(inputs, outputs, (["state_in_0", "serving_default_state_in_0:0"], ["c", "d"]))


class FakeInterpreter:
    """A LiteRT stand-in: input 0 is float32 data [1, 3]; input 1 is int8 state [1, 3], tied to output 1
    of shape [1, 1, 3] (so the carry needs a reshape); output 0 is data + state as float32."""

    def __init__(self, signature=True):
        q = (0.5, -3)
        self.inputs = [
            {
                "name": "serving_default_x:0",
                "index": 0,
                "shape": np.array([1, 3]),
                "dtype": np.float32,
                "quantization": (0.0, 0),
            },
            {"name": "serving_default_s:0", "index": 1, "shape": np.array([1, 3]), "dtype": np.int8, "quantization": q},
        ]
        self.outputs = [
            {
                "name": "StatefulPartitionedCall:0",
                "index": 2,
                "shape": np.array([1, 3]),
                "dtype": np.float32,
                "quantization": (0.0, 0),
            },
            {
                "name": "StatefulPartitionedCall:1",
                "index": 3,
                "shape": np.array([1, 1, 3]),
                "dtype": np.int8,
                "quantization": q,
            },
        ]
        self.signature, self.values = signature, {}

    def get_input_details(self):
        return self.inputs

    def get_output_details(self):
        return self.outputs

    def set_tensor(self, index, value):
        detail = self.inputs[index]
        if value.shape != tuple(detail["shape"]) or value.dtype != detail["dtype"]:
            raise ValueError(f"tensor {index}: got {value.shape} {value.dtype}")
        self.values[index] = value.copy()

    def invoke(self):
        state = self.values[1].astype(np.int16)
        self.values[2] = self.values[0] + state.astype(np.float32)
        self.values[3] = np.clip(state + 1, -128, 127).astype(np.int8).reshape(1, 1, 3)

    def get_tensor(self, index):
        return self.values[index]

    def get_signature_list(self):
        return {"serving_default": {}} if self.signature else {}

    def get_signature_runner(self, key):
        interpreter = self

        class Runner:
            def get_input_details(self):
                return {"x": interpreter.inputs[0], "state_in_0": interpreter.inputs[1]}

            def get_output_details(self):
                return {"y": interpreter.outputs[0], "state_out_0": interpreter.outputs[1]}

        return Runner()


@pytest.fixture
def fake(monkeypatch):
    from helia_model_zoo import runtime

    monkeypatch.setattr(runtime, "interpreter", lambda model, resolver=None: FakeInterpreter())
    inputs, outputs = runtime.model_tensors("fake.tflite")
    return inputs, outputs


def test_signature_names_pair_state(fake):
    from helia_model_zoo import runtime

    inputs, outputs = fake
    assert runtime.signature_names("fake.tflite") == (["x", "state_in_0"], ["y", "state_out_0"])
    assert state_pairs_from_names(inputs, outputs, runtime.signature_names("fake.tflite")) == (
        StatePair("serving_default_s:0", "StatefulPartitionedCall:1"),
    )


def test_sequence_carry_reshapes_and_floats_are_drawn_as_floats(fake, tmp_path):
    from helia_model_zoo.manifest import Precision

    inputs, outputs = fake
    precision = Precision("fake", FileRef("repo://fake.tflite"), inputs, outputs, None)
    io = IO("explicit_state", (StatePair("serving_default_s:0", "StatefulPartitionedCall:1"),))
    arrays = generate("fake.tflite", precision, io, kind="sequence", steps=4, resets=[2], seed=0)
    assert arrays["input_1"][:, 0, 0].tolist() == [-3, -2, -3, -2]
    assert not np.all(arrays["input_0"] == np.round(arrays["input_0"]))
    assert problems(arrays, precision, io, "sequence", 4, [2]) == []
    path = tmp_path / "g.npz"
    golden.write(path, arrays)
    assert check("fake.tflite", path, kind="sequence", steps=4, resets=[2]) == []


def test_cli_record_block_records_resolver_and_runtime(root, tmp_path, capsys):
    from importlib import metadata

    out = tmp_path / "g.npz"
    assert main(["golden", "generate", str(root / RNNOISE), str(out), "--print-record"]) == 0
    block = json.loads(capsys.readouterr().out)
    assert (block["resolver"], block["runtime"]) == (
        "builtin_ref",
        f"ai-edge-litert=={metadata.version('ai-edge-litert')}",
    )
    assert main(["golden", "generate", str(root / RNNOISE), str(out), "--print-record", "--resolver", "builtin"]) == 0
    assert json.loads(capsys.readouterr().out)["resolver"] == "builtin"


def test_drawn_inputs_match_the_original_generator(rnnoise):
    model, precision, _ = rnnoise
    arrays = generate(model, precision, IO("stateless", ()), seed=7)
    rng = np.random.default_rng(7)
    for i, tensor in enumerate(precision.inputs):
        assert np.array_equal(arrays[f"input_{i}"], rng.integers(-8, 8, size=tensor.shape, dtype=np.int8))


def test_cli_reports_bad_pairs_and_missing_files(root, tmp_path, capsys):
    model = str(root / DFNET2)
    out = str(tmp_path / "g.npz")
    assert main(["golden", "generate", model, out, "--kind", "sequence", "--steps", "2", "--pair", "9:9"]) == 1
    assert "pair 9:9 is out of range for 3 inputs and 5 outputs" in capsys.readouterr().err
    assert main(["golden", "generate", model, out, "--pair", "²:1"]) == 1
    assert "--pair expects IN:OUT" in capsys.readouterr().err
    assert main(["golden", "generate", model, out, "--data", str(tmp_path / "missing.npz")]) == 1
    assert "missing.npz" in capsys.readouterr().err
    assert main(["golden", "check", model, str(tmp_path / "missing.npz")]) == 2
    assert "missing.npz" in capsys.readouterr().err
