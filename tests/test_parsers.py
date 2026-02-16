from pathlib import Path

from stagpy import stagyyparsers as prs
from stagpy.stagyydata import StagyyData


def test_time_series_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    data = prs.time_series(sdat.filename("time.dat"))
    assert data is not None
    assert (data.columns[3:6] == ["Tmin", "Tmean", "Tmax"]).all()


def test_time_series_h5(sdat_h5: StagyyData) -> None:
    sdat = sdat_h5
    assert sdat.hdf5 is not None
    data = prs.time_series_h5(sdat.hdf5 / "TimeSeries.h5")
    assert data is not None
    assert (data.columns[3:6] == ["Tmin", "Tmean", "Tmax"]).all()


def test_time_series_invalid_prs() -> None:
    assert prs.time_series(Path("dummy")) is None


def test_rprof_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    data, time = prs.rprof(sdat.filename("rprof.dat"))
    assert all((df.columns[:3] == ["r", "Tmean", "Tmin"]).all() for df in data.values())


def test_rprof_h5(sdat_h5: StagyyData) -> None:
    sdat = sdat_h5
    assert sdat.hdf5 is not None
    data, _times = prs.rprof_h5(sdat.hdf5 / "rprof.h5")
    assert data is not None
    assert (data[1000].columns[:3] == ["r", "Tmean", "Tmin"]).all()


def test_rprof_invalid_prs() -> None:
    assert prs.rprof(Path("dummy")) == ({}, None)


def test_fields_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    parsed = prs.fields(sdat.filename("t", len(sdat.snaps) - 1))
    assert parsed is not None
    hdr, flds = parsed
    assert flds.shape[0] == 1
    assert flds.shape[4] == 1
    assert flds.shape[1:4] == tuple(hdr["nts"])


def test_field_header_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    hdr = prs.field_header(sdat.filename("t", len(sdat.snaps) - 1))
    assert hdr is not None
    assert hdr["nts"].shape == (3,)


def test_fields_istep_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    istep = prs.field_istep(sdat.filename("t", len(sdat.snaps) - 1))
    assert istep == sdat.snaps[-1].istep


def test_fields_invalid_prs() -> None:
    assert prs.fields(Path("dummy")) is None


def test_refstate_parser(example_h5_path: Path) -> None:
    out = prs.refstate(example_h5_path / "output_refstat.dat")
    cols = ["z", "T", "rho", "expan", "Cp", "Tcond"]
    assert out is not None
    systems, adias = out
    assert (systems[0][0].columns == cols).all()
    assert (adias[0].columns == cols).all()
