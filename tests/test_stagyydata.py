from itertools import zip_longest as zlong
import pytest
import f90nml
import pandas
import stagpy.stagyydata

def test_sdat_path(example_dir, sdat):
    assert sdat.path == example_dir

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
    assert all(a is b for a, b in zlong(sdat.steps, sdat.steps[:]))

def test_snaps_iter(sdat):
    assert all(a is b for a, b in zlong(sdat.snaps, sdat.snaps[:]))

def test_filter_snap(sdat):
    snaps = (s for s in sdat.snaps if s)
    assert all(
        a is b for a, b in zlong(snaps, sdat.steps.filter(snap=True)))

def test_filter_even_steps(sdat):
    even = lambda step: step.istep % 2 == 0
    assert all(
        a is b for a,b in zlong(sdat.steps[::2], sdat.steps.filter(func=even)))

def test_snaps_last(sdat):
    assert sdat.snaps.last is sdat.snaps[-1]

def test_snaps_empty(sdat):
    empty = sdat.snaps[len(sdat.snaps)]
    assert isinstance(empty, stagpy.stagyydata._EmptyStep)

def test_step_is_snap(sdat):
    istep = sdat.snaps.last.istep
    assert sdat.steps[istep] is sdat.snaps.last

def test_step_sdat(sdat):
    assert all(s.sdat is sdat for s in sdat.steps)

def test_step_istep(sdat):
    assert all(s is sdat.steps[s.istep] for s in sdat.steps)

def test_geom_refs(step):
    assert step.geom is step.fields.geom

def test_geom_twod(step):
    assert step.geom.twod
