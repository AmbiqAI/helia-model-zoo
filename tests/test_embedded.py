# SPDX-FileCopyrightText: 2026 Ambiq AI
# SPDX-License-Identifier: BSD-3-Clause
import hashlib

import pytest

from helia_model_zoo.cli import main

MODEL = b"\x1c\0\0\0TFL3"


def header(model=MODEL):
    return (
        b"unsigned char other[] = {0xff};\nNS_PUT_IN_TCM unsigned char mut_model[] __attribute__((aligned(16))) = {"
        + b",".join(f"0x{v:02x}".encode() for v in model)
        + b",};\nunsigned int length = 8;"
    )


def command(source, output, content):
    return [
        "extract-header",
        "--header",
        str(source),
        "--header-sha256",
        hashlib.sha256(content).hexdigest(),
        "--header-bytes",
        str(len(content)),
        "--symbol",
        "mut_model",
        "--sha256",
        hashlib.sha256(MODEL).hexdigest(),
        "--bytes",
        str(len(MODEL)),
        "--output",
        str(output),
    ]


def test_cli_extracts_only_the_verified_named_model(tmp_path):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    assert main(command(source, output, content)) == 0
    assert output.read_bytes() == MODEL


@pytest.mark.parametrize(
    "option,value",
    [
        ("--header-sha256", "0" * 64),
        ("--header-bytes", "1"),
        ("--sha256", "0" * 64),
        ("--bytes", "1"),
        ("--header-sha256", "../wrong"),
        ("--bytes", "-1"),
        ("--symbol", "missing"),
    ],
)
def test_cli_bad_source_or_model_identity_leaves_no_output(tmp_path, option, value):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    args = command(source, output, content)
    args[args.index(option) + 1] = value
    assert main(args) == 1
    assert not output.exists()


def test_cli_missing_header_and_existing_output_are_safe(tmp_path):
    source = tmp_path / "model.h"
    output = tmp_path / "model.tflite"
    assert main(command(source, output, header())) == 1
    source.write_bytes(header())
    output.write_bytes(b"retained")
    assert main(command(source, output, header())) == 1
    assert output.read_bytes() == b"retained"


def test_cli_corrupted_header_cannot_create_output(tmp_path):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content.replace(b"other", b"wrong"))
    output = tmp_path / "model.tflite"
    assert main(command(source, output, content)) == 1
    assert not output.exists()


@pytest.mark.parametrize("body", [b"0x100", b"-1", b"256", b"012", b"1 << 4", b"MACRO", b"1,,2"])
def test_extraction_refuses_expressions_and_non_byte_literals(body):
    from helia_model_zoo.embedded import extract_c_array

    with pytest.raises(ValueError):
        extract_c_array(b"unsigned char model[] = {" + body + b"};", "model")


def test_array_comments_and_declared_length():
    from helia_model_zoo.embedded import extract_c_array

    content = b"const uint8_t model[8] = {28,/* 999 */0,0,0,84,70,// 999\n76,51,};"
    assert extract_c_array(content, "model") == MODEL
    with pytest.raises(ValueError, match="declared extent"):
        extract_c_array(content.replace(b"[8]", b"[9]"), "model")


def test_duplicate_symbol_and_non_model_bytes_are_refused():
    from helia_model_zoo.embedded import extract_c_array

    with pytest.raises(ValueError, match="exactly one"):
        extract_c_array(header() + header(), "mut_model")
    with pytest.raises(ValueError, match="TFLite"):
        extract_c_array(header(b"other"), "mut_model")
    with pytest.raises(ValueError, match="identifier"):
        extract_c_array(header(), "mut_.*")


def test_extraction_does_not_load_inference_runtime():
    import subprocess
    import sys

    code = (
        "import sys; from helia_model_zoo.embedded import extract_c_array; "
        "assert extract_c_array(b\"unsigned char m[]={28,0,0,0,84,70,76,51};\",'m') == b'\\x1c\\0\\0\\0TFL3'; "
        "assert not any(m in sys.modules for m in ['ai_edge_litert','tensorflow','helia_aot','helia_profiler'])"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
