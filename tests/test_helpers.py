import pytest

from stagpy import _helpers
from stagpy.config import Config
from stagpy.stagyydata import StagyyData


def test_walk_dflt(sdat: StagyyData) -> None:
    view = _helpers.walk(sdat, Config.default_())
    wlk = iter(view)
    assert next(wlk) is sdat.snaps[-1]
    with pytest.raises(StopIteration):
        next(wlk)


def test_out_name_conf() -> None:
    conf = Config.default_()
    oname = "something_fancy"
    conf.core.outname = oname
    stem = "teapot"
    assert _helpers.out_name(conf, stem) == oname + "_" + stem


def test_out_name_number() -> None:
    conf = Config.default_()
    assert _helpers.out_name(conf, "T", 123) == "stagpy_T00123"


def test_baredoc() -> None:
    """
       Badly formatted docstring .. .

    With some content.

    """
    expected = "Badly formatted docstring"
    assert _helpers.baredoc(test_baredoc) == expected
