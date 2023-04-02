import pytest

import stagpy.error
import stagpy.phyvars
import stagpy.rprof


def test_no_rprof_data(sdat):
    with pytest.raises(stagpy.error.MissingDataError):
        sdat.steps[1].rprofs["Tmean"]


def test_invalid_rprof(step):
    with pytest.raises(stagpy.error.UnknownRprofVarError):
        step.rprofs["DummyVar"]


def test_rprof_bounds_if_no_rprofs(sdat):
    rcmb, rtot = sdat.steps[1].rprofs.bounds
    assert rtot > rcmb


def test_get_rprof(step):
    rpf = step.rprofs["Tmean"]
    assert rpf.rad is step.rprofs.centers
    assert rpf.values.shape == (step.geom.nztot,)
    assert rpf.meta == stagpy.phyvars.RPROF["Tmean"]


def test_get_rprof_extra(step):
    rpf = step.rprofs["diff"]
    assert rpf.rad is step.rprofs.walls
    assert rpf.values.shape == rpf.rad.shape
    assert isinstance(rpf.meta, stagpy.phyvars.Varr)
