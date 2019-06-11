import re
import stagpy.commands


def test_info_cmd(capsys, example_dir):
    stagpy.conf.core.path = example_dir
    stagpy.commands.info_cmd()
    output = capsys.readouterr()
    expected = re.compile(
        r'^StagYY run in.*\n.* x .*\n\nStep.*, snapshot.*\n  t.*\n\n$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.out)
    del stagpy.conf.core.path


def test_var_cmd(capsys):
    stagpy.commands.var_cmd()
    output = capsys.readouterr()
    expected = re.compile(
        r'field:\n.*\nrprof:\n.*\ntime:\n.*\nplates:\n.*$',
        flags=re.DOTALL)
    assert expected.fullmatch(output.out)


def test_version_cmd(capsys):
    stagpy.commands.version_cmd()
    output = capsys.readouterr()
    expected = 'stagpy version: {}\n'.format(stagpy.__version__)
    assert output.out == expected


def test_config_cmd(capsys):
    stagpy.commands.config_cmd()
    output = capsys.readouterr()
    expected = '(c|f): available only as CLI argument/in the config file'
    assert output.out.startswith(expected)
