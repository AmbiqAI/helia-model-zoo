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


@pytest.mark.parametrize("body", [b"0x100", b"-1", b"256", b"1 << 4", b"MACRO", b"1,,2"])
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


@pytest.mark.parametrize(
    "content",
    [
        b'const char *help = "unsigned char mut_model[]={28,0,0,0,84,70,76,51};";',
        b"// continued comment \\\nunsigned char mut_model[]={28,0,0,0,84,70,76,51};\n",
        b"#if 0\nunsigned char mut_model[]={28,0,0,0,84,70,76,51};\n#else\n"
        b"typedef unsigned char Byte; Byte mut_model[]={1};\n#endif\n",
    ],
)
def test_cli_refuses_lexical_decoys_and_conditional_arrays(tmp_path, content):
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    assert main(command(source, output, content)) == 1
    assert not output.exists()


def test_string_comment_markers_do_not_hide_a_real_array():
    from helia_model_zoo.embedded import extract_c_array

    content = b'const char *before = "/*";\n' + header() + b'\nconst char *after = "*/";'
    assert extract_c_array(content, "mut_model") == MODEL


def test_comments_do_not_join_integer_tokens():
    from helia_model_zoo.embedded import extract_c_array

    with pytest.raises(ValueError):
        extract_c_array(b"unsigned char mut_model[]={28,0,0,0,8/**/4,70,76,51};", "mut_model")


def test_c_octal_extent_and_canonical_header_guard():
    from helia_model_zoo.embedded import extract_c_array

    content = b"#ifndef MODEL_H\n#define MODEL_H\nuint8_t model[010]={034,0,0,0,0124,0106,0114,063};\n#endif\n"
    assert extract_c_array(content, "model") == MODEL


def test_cli_absurd_header_size_is_a_source_error(tmp_path):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    args = command(source, output, content)
    args[args.index("--header-bytes") + 1] = str(2**100)
    assert main(args) == 1
    assert not output.exists()


@pytest.mark.parametrize(
    "directive",
    [b"\v#", b"\f#", b"#\v", b"#\f", b"%:", b"??=", b"\v%:"],
)
def test_cli_rejects_all_c_directive_spellings(tmp_path, directive):
    content = directive + b"if 0\n" + header() + b"\n" + directive + b"endif\n"
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    assert main(command(source, output, content)) == 1
    assert not output.exists()


@pytest.mark.parametrize("prefix,suffix", [(b"DROP(\n", b"\n)"), (b"void f(){\n", b"\n}"), (b"void f()<%\n", b"\n%>")])
def test_macro_argument_or_block_is_not_a_global_array(tmp_path, prefix, suffix):
    content = b"#define DROP(...)\n" + prefix + header() + suffix
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    assert main(command(source, output, content)) == 1
    assert not output.exists()


def test_small_header_with_large_expected_size_does_not_allocate(tmp_path, monkeypatch):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    args = command(source, output, content)
    args[args.index("--header-bytes") + 1] = str(1024**3)
    real_open = type(source).open

    class NoLargeRead:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            raise AssertionError("mismatched header size must be rejected before reading")

    monkeypatch.setattr(type(source), "open", lambda p, *a, **kw: NoLargeRead() if p == source else real_open(p, *a, **kw))
    assert main(args) == 1
    assert not output.exists()


def test_cli_read_memory_failure_is_a_clean_refusal(tmp_path, monkeypatch):
    content = header()
    source = tmp_path / "model.h"
    source.write_bytes(content)
    output = tmp_path / "model.tflite"
    real_open = type(source).open

    class FailedRead:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            raise MemoryError("header allocation failed")

    monkeypatch.setattr(type(source), "open", lambda p, *a, **kw: FailedRead() if p == source else real_open(p, *a, **kw))
    assert main(command(source, output, content)) == 1
    assert not output.exists()


def test_trigraph_line_splicing_and_cr_only_lines_are_lexical_boundaries():
    from helia_model_zoo.embedded import extract_c_array

    with pytest.raises(ValueError):
        extract_c_array(b"// continued??/\n" + header().replace(b"\n", b" "), "mut_model")
    with pytest.raises(ValueError):
        extract_c_array(b"#if 0\r" + header() + b"\r#endif\r", "mut_model")


def test_extraction_does_not_load_inference_runtime():
    import subprocess
    import sys

    code = (
        "import sys; from helia_model_zoo.embedded import extract_c_array; "
        "assert extract_c_array(b\"unsigned char m[]={28,0,0,0,84,70,76,51};\",'m') == b'\\x1c\\0\\0\\0TFL3'; "
        "assert not any(m in sys.modules for m in ['ai_edge_litert','tensorflow','helia_aot','helia_profiler'])"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
