import re

import pytest
from pytest import CaptureFixture

import stagpy.args


def test_no_args(capsys: CaptureFixture) -> None:
    stagpy.args.parse_args([])()
    output = capsys.readouterr()
    expected = re.compile(
        r"StagPy is a tool to.*" r"Run `stagpy -h` for usage\n$", flags=re.DOTALL
    )
    assert expected.fullmatch(output.out)


def test_help(capsys: CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(["-h"])
    output = capsys.readouterr()
    expected = re.compile(
        r"^usage:.*\nStagPy is a tool to.*\n"
        r".*field.*Plot scalar and vector fields\n"
        r".*--help.*show this help message and exit.*$",
        flags=re.DOTALL,
    )
    assert expected.fullmatch(output.out)


def test_invalid_argument(capsys: CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(["-dummyinvalidarg"])
    output = capsys.readouterr()
    expected = re.compile(
        r"^usage: .*error: unrecognized arguments:.*\n$", flags=re.DOTALL
    )
    assert expected.fullmatch(output.err)


def test_invalid_subcmd(capsys: CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(["dummyinvalidcmd"])
    output = capsys.readouterr()
    expected = re.compile(r"^usage: .*error:.*invalid choice:.*\n$", flags=re.DOTALL)
    assert expected.fullmatch(output.err)


def test_field_subcmd() -> None:
    func = stagpy.args.parse_args(["field"])
    assert func is stagpy.field.cmd


def test_rprof_subcmd() -> None:
    func = stagpy.args.parse_args(["rprof"])
    assert func is stagpy.rprof.cmd


def test_time_cmd() -> None:
    func = stagpy.args.parse_args(["time"])
    assert func is stagpy.time_series.cmd


def test_plates_subcmd() -> None:
    func = stagpy.args.parse_args(["plates"])
    assert func is stagpy.plates.cmd


def test_info_subcmd() -> None:
    func = stagpy.args.parse_args(["info"])
    assert func is stagpy.commands.info_cmd


def test_var_subcmd() -> None:
    func = stagpy.args.parse_args(["var"])
    assert func is stagpy.commands.var_cmd


def test_version_subcmd() -> None:
    func = stagpy.args.parse_args(["version"])
    assert func is stagpy.commands.version_cmd


def test_config_subcmd() -> None:
    func = stagpy.args.parse_args(["config"])
    assert func is stagpy.commands.config_cmd
