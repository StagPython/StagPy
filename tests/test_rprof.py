import pytest
import stagpy.error
import stagpy.rprof
import stagpy.phyvars


def test_no_rprof_data(sdat):
    with pytest.raises(stagpy.error.MissingDataError):
        sdat.steps[1].rprofs['Tmean']


def test_invalid_rprof(step):
    with pytest.raises(stagpy.error.UnknownRprofVarError):
        step.rprofs['DummyVar']


def test_rprof_bounds_if_no_rprofs(sdat):
    rcmb, rtot = sdat.steps[1].rprofs.bounds
    assert rtot > rcmb


def test_get_rprof(step):
    prof, rad, meta = step.rprofs['Tmean']
    assert rad is step.rprofs.centers
    assert prof.shape == (step.geom.nztot,)
    assert meta == stagpy.phyvars.RPROF['Tmean']


def test_get_rprof_extra(step):
    prof, rad, meta = step.rprofs['diff']
    assert rad is step.rprofs.walls
    assert prof.shape == rad.shape
    assert isinstance(meta, stagpy.phyvars.Varr)
