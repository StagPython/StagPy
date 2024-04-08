import pytest

import stagpy
import stagpy._helpers
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
    oname = "something_fancy"
    stagpy.conf.core.outname = oname
    stem = "teapot"
    assert stagpy._helpers.out_name(stem) == oname + "_" + stem
    del stagpy.conf.core.outname


def test_out_name_number() -> None:
    assert stagpy._helpers.out_name("T", 123) == "stagpy_T00123"


def test_baredoc() -> None:
    """
       Badly formatted docstring .. .

    With some content.

    """
    expected = "Badly formatted docstring"
    assert stagpy._helpers.baredoc(test_baredoc) == expected
