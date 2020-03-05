import pytest
import f90nml
import pandas
import stagpy.stagyydata
import stagpy._step
import stagpy.error


def test_sdat_path(example_dir, sdat):
    assert sdat.path == example_dir


def test_sdat_deflt_nfields_max(sdat):
    assert sdat.nfields_max == 50


def test_sdat_set_nfields_max(sdat):
    sdat.nfields_max = 6
    assert sdat.nfields_max == 6


def test_sdat_set_nfields_max_none(sdat):
    sdat.nfields_max = None
    assert sdat.nfields_max is None


def test_sdat_set_nfields_max_invalid(sdat):
    with pytest.raises(stagpy.error.InvalidNfieldsError) as err:
        sdat.nfields_max = 5
    assert err.value.nfields == 5


def test_sdat_par(sdat):
    assert isinstance(sdat.par, f90nml.namelist.Namelist)


def test_sdat_tseries(sdat):
    assert isinstance(sdat.tseries, pandas.DataFrame)


def test_sdat_rprof(sdat):
    assert isinstance(sdat.rprof, pandas.DataFrame)


def test_sdat_walk_dflt(sdat):
    wlk = iter(sdat.walk)
    assert next(wlk) is sdat.snaps[-1]
    with pytest.raises(StopIteration):
        next(wlk)


def test_steps_iter(sdat):
    assert sdat.steps[:] == sdat.steps


def test_snaps_iter(sdat):
    assert sdat.snaps[:] == sdat.snaps


def test_filter_snap(sdat):
    snaps = (s for s in sdat.snaps if s)
    assert sdat.steps.filter(snap=True) == snaps


def test_filter_even_steps(sdat):
    even = sdat.steps.filter(func=lambda step: step.istep % 2 == 0)
    assert sdat.steps[::2] == even


def test_snaps_last(sdat):
    assert sdat.snaps[-1] is sdat.snaps[len(sdat.snaps) - 1]


def test_snaps_empty(sdat):
    with pytest.raises(stagpy.error.InvalidSnapshotError):
        sdat.snaps[len(sdat.snaps)]


def test_step_is_snap(sdat):
    istep = sdat.snaps[-1].istep
    assert sdat.steps[istep] is sdat.snaps[-1]


def test_step_sdat(sdat):
    assert all(s.sdat is sdat for s in sdat.steps)


def test_step_istep(sdat):
    assert all(s is sdat.steps[s.istep] for s in sdat.steps)


def test_geom_refs(step):
    assert step.geom is step.fields.geom


def test_geom_twod(step):
    assert step.geom.twod
