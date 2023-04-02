import pytest

import stagpy.error
import stagpy.phyvars
import stagpy.rprof
from stagpy.stagyydata import StagyyData, Step


def test_no_rprof_data(sdat: StagyyData) -> None:
    with pytest.raises(stagpy.error.MissingDataError):
        sdat.steps[1].rprofs["Tmean"]


def test_invalid_rprof(step: Step) -> None:
    with pytest.raises(stagpy.error.UnknownRprofVarError):
        step.rprofs["DummyVar"]


def test_rprof_bounds_if_no_rprofs(sdat: StagyyData) -> None:
    rcmb, rtot = sdat.steps[1].rprofs.bounds
    assert rtot > rcmb


def test_get_rprof(step: Step) -> None:
    rpf = step.rprofs["Tmean"]
    assert rpf.rad is step.rprofs.centers
    assert rpf.values.shape == (step.geom.nztot,)
    assert rpf.meta == stagpy.phyvars.RPROF["Tmean"]


def test_get_rprof_extra(step: Step) -> None:
    rpf = step.rprofs["diff"]
    assert rpf.rad is step.rprofs.walls
    assert rpf.values.shape == rpf.rad.shape
