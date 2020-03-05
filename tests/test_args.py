import re
import pytest
import stagpy.args


def test_no_args(capsys):
    stagpy.args.parse_args([])()
    output = capsys.readouterr()
    expected = re.compile(
        r'StagPy is a tool to.*'
        r'Run `stagpy -h` for usage\n$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.out)


def test_help(capsys):
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(['-h'])
    output = capsys.readouterr()
    expected = re.compile(
        r'^usage:.*\nStagPy is a tool to.*'
        r'\npositional arguments.*\noptional arguments.*$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.out)


def test_invalid_argument(capsys):
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(['-dummyinvalidarg'])
    output = capsys.readouterr()
    expected = re.compile(
        r'^usage: .*error: unrecognized arguments:.*\n$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.err)


def test_invalid_subcmd(capsys):
    with pytest.raises(SystemExit):
        stagpy.args.parse_args(['dummyinvalidcmd'])
    output = capsys.readouterr()
    expected = re.compile(
        r'^usage: .*error:.*invalid choice:.*\n$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.err)


def test_field_subcmd():
    func = stagpy.args.parse_args(['field'])
    assert func is stagpy.field.cmd


def test_rprof_subcmd():
    func = stagpy.args.parse_args(['rprof'])
    assert func is stagpy.rprof.cmd


def test_time_cmd():
    func = stagpy.args.parse_args(['time'])
    assert func is stagpy.time_series.cmd


def test_plates_subcmd():
    func = stagpy.args.parse_args(['plates'])
    assert func is stagpy.plates.cmd


def test_info_subcmd():
    func = stagpy.args.parse_args(['info'])
    assert func is stagpy.commands.info_cmd


def test_var_subcmd():
    func = stagpy.args.parse_args(['var'])
    assert func is stagpy.commands.var_cmd


def test_version_subcmd():
    func = stagpy.args.parse_args(['version'])
    assert func is stagpy.commands.version_cmd


def test_config_subcmd():
    func = stagpy.args.parse_args(['config'])
    assert func is stagpy.commands.config_cmd
