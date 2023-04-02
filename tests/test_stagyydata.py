from pathlib import Path

import pytest

import stagpy._step
import stagpy.error
from stagpy.stagyydata import StagyyData, Step


def test_sdat_path(example_dir: Path, sdat: StagyyData) -> None:
    assert sdat.path == example_dir


def test_sdat_deflt_nfields_max(sdat: StagyyData) -> None:
    assert sdat.nfields_max == 50


def test_sdat_set_nfields_max(sdat: StagyyData) -> None:
    sdat.nfields_max = 6
    assert sdat.nfields_max == 6


def test_sdat_set_nfields_max_none(sdat: StagyyData) -> None:
    sdat.nfields_max = None
    assert sdat.nfields_max is None


def test_sdat_set_nfields_max_invalid(sdat: StagyyData) -> None:
    with pytest.raises(stagpy.error.InvalidNfieldsError) as err:
        sdat.nfields_max = 5
    assert err.value.nfields == 5


def test_sdat_tseries(sdat: StagyyData) -> None:
    assert sdat.tseries["Tmean"].time is sdat.tseries.time


def test_sdat_walk_dflt(sdat: StagyyData) -> None:
    wlk = iter(sdat.walk)
    assert next(wlk) is sdat.snaps[-1]
    with pytest.raises(StopIteration):
        next(wlk)


def test_steps_iter(sdat: StagyyData) -> None:
    assert sdat.steps[:] == sdat.steps


def test_snaps_iter(sdat: StagyyData) -> None:
    assert sdat.snaps[:] == sdat.snaps


def test_filter_snap(sdat: StagyyData) -> None:
    snaps = (s for s in sdat.snaps if s)
    assert sdat.steps.filter(snap=True) == snaps


def test_filter_even_steps(sdat: StagyyData) -> None:
    even = sdat.steps.filter(func=lambda step: step.istep % 2 == 0)
    assert sdat.steps[::2] == even


def test_snaps_last(sdat: StagyyData) -> None:
    assert sdat.snaps[-1] is sdat.snaps[len(sdat.snaps) - 1]


def test_snaps_empty(sdat: StagyyData) -> None:
    with pytest.raises(stagpy.error.InvalidSnapshotError):
        sdat.snaps[len(sdat.snaps)]


def test_step_is_snap(sdat: StagyyData) -> None:
    istep = sdat.snaps[-1].istep
    assert sdat.steps[istep] is sdat.snaps[-1]


def test_step_sdat(sdat: StagyyData) -> None:
    assert all(s.sdat is sdat for s in sdat.steps)


def test_step_istep(sdat: StagyyData) -> None:
    assert all(s is sdat.steps[s.istep] for s in sdat.steps)


def test_geom_refs(step: Step) -> None:
    assert step.geom is step.fields.geom


def test_geom(step: Step) -> None:
    assert step.geom.twod
    assert not step.geom.threed
    assert not step.geom.yinyang
    assert step.geom.cartesian is not step.geom.spherical
