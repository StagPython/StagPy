from pkg_resources import safe_version
import stagpy


def test_version_format():
    assert stagpy.__version__ == safe_version(stagpy.__version__)


def test_bogus_mplstyle(capsys):
    stagpy.conf.plot.mplstyle = 'stagpy-bogus'
    stagpy.load_mplstyle()
    output = capsys.readouterr()
    assert output.err == 'Cannot import style stagpy-bogus.\n'
    del stagpy.conf.plot.mplstyle
