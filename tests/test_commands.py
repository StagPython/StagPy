import re
from pathlib import Path

from pytest import CaptureFixture

from stagpy import __version__, commands
from stagpy.config import Config


def test_info_cmd(capsys: CaptureFixture, example_dir: Path) -> None:
    conf = Config.default_()
    conf.core.path = example_dir
    commands.info_cmd(conf)
    output = capsys.readouterr()
    expected = re.compile(
        r"^StagYY run in.*\n.* x .*\n\nStep.*, snapshot.*\n  t.*\n\n$", flags=re.DOTALL
    )
    assert expected.fullmatch(output.out)


def test_version_cmd(capsys: CaptureFixture) -> None:
    commands.version_cmd(Config.default_())
    output = capsys.readouterr()
    expected = "stagpy version: {}\n".format(__version__)
    assert output.out == expected
