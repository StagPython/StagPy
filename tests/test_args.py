import re

import pytest
from pytest import CaptureFixture

import stagpy
from stagpy.args import parse_args
from stagpy.config import Config


def test_no_args(capsys: CaptureFixture) -> None:
    conf = Config.default_()
    parse_args(conf, [])(conf)
    output = capsys.readouterr()
    expected = re.compile(
        r"StagPy is a tool to.*" r"Run `stagpy -h` for usage\n$", flags=re.DOTALL
    )
    assert expected.fullmatch(output.out)


def test_help(capsys: CaptureFixture) -> None:
    conf = Config.default_()
    with pytest.raises(SystemExit):
        parse_args(conf, ["-h"])
    output = capsys.readouterr()
    expected = re.compile(
        r"^usage:.*\nStagPy is a tool to.*\n"
        r".*field.*Plot scalar and vector fields\n"
        r".*--help.*show this help message and exit.*$",
        flags=re.DOTALL,
    )
    assert expected.fullmatch(output.out)


def test_invalid_argument(capsys: CaptureFixture) -> None:
    conf = Config.default_()
    with pytest.raises(SystemExit):
        parse_args(conf, ["-dummyinvalidarg"])
    output = capsys.readouterr()
    expected = re.compile(
        r"^usage: .*error: unrecognized arguments:.*\n$", flags=re.DOTALL
    )
    assert expected.fullmatch(output.err)


def test_invalid_subcmd(capsys: CaptureFixture) -> None:
    conf = Config.default_()
    with pytest.raises(SystemExit):
        parse_args(conf, ["dummyinvalidcmd"])
    output = capsys.readouterr()
    expected = re.compile(r"^usage: .*error:.*invalid choice:.*\n$", flags=re.DOTALL)
    assert expected.fullmatch(output.err)


def test_field_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["field"])
    assert func is stagpy.field.cmd


def test_rprof_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["rprof"])
    assert func is stagpy.rprof.cmd


def test_time_cmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["time"])
    assert func is stagpy.time_series.cmd


def test_plates_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["plates"])
    assert func is stagpy.plates.cmd


def test_info_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["info"])
    assert func is stagpy.commands.info_cmd


def test_var_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["var"])
    assert func is stagpy.commands.var_cmd


def test_version_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["version"])
    assert func is stagpy.commands.version_cmd


def test_config_subcmd() -> None:
    conf = Config.default_()
    func = parse_args(conf, ["config"])
    assert func is stagpy.commands.config_cmd
