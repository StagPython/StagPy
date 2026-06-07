from pathlib import Path

from stagpy import parsers
from stagpy.stagyydata import StagyyData


def test_time_series_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    data = parsers.txt.tseries(sdat.par.legacy_output("time.dat"))
    assert data is not None
    assert (data.columns[3:6] == ["Tmin", "Tmean", "Tmax"]).all()


def test_time_series_h5(sdat_h5: StagyyData) -> None:
    path = sdat_h5.par.h5_output("TimeSeries.h5")
    data = parsers.h5.tseries.tseries(path)
    assert data is not None
    assert (data.columns[3:6] == ["Tmin", "Tmean", "Tmax"]).all()


def test_time_series_invalid_prs() -> None:
    assert parsers.txt.tseries(Path("dummy")) is None


def test_rprof_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    data, _time = parsers.txt.rprof(sdat.par.legacy_output("rprof.dat"))
    assert all((df.columns[:3] == ["r", "Tmean", "Tmin"]).all() for df in data.values())


def test_rprof_h5(sdat_h5: StagyyData) -> None:
    path = sdat_h5.par.h5_output("rprof.h5")
    data, _times = parsers.h5.rprof.rprof(path)
    assert data is not None
    assert (data[1000].columns[:3] == ["r", "Tmean", "Tmin"]).all()


def test_rprof_invalid_prs() -> None:
    assert parsers.txt.rprof(Path("dummy")) == ({}, None)


def test_fields_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    parsed = parsers.bin.field.field(sdat.par.legacy_output("t", len(sdat.snaps) - 1))
    assert parsed is not None
    hdr, flds = parsed
    assert flds.shape[0] == 1
    assert flds.shape[4] == 1
    assert flds.shape[1:4] == tuple(hdr["nts"])


def test_field_header_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    hdr = parsers.bin.field.header(sdat.par.legacy_output("t", len(sdat.snaps) - 1))
    assert hdr is not None
    assert hdr["nts"].shape == (3,)


def test_fields_istep_prs(sdat_legacy: StagyyData) -> None:
    sdat = sdat_legacy
    istep = parsers.bin.field.istep(sdat.par.legacy_output("t", len(sdat.snaps) - 1))
    assert istep == sdat.snaps[-1].istep


def test_fields_invalid_prs() -> None:
    assert parsers.bin.field.field(Path("dummy")) is None


def test_refstate_parser(example_h5_path: Path) -> None:
    out = parsers.txt.refstate(example_h5_path / "output_refstat.dat")
    cols = ["z", "T", "rho", "expan", "Cp", "Tcond"]
    assert out is not None
    systems, adias = out
    assert (systems[0][0].columns == cols).all()
    assert (adias[0].columns == cols).all()
