import stagpy.phyvars
import stagpy.time_series
from stagpy.stagyydata import StagyyData


def test_get_time_series(sdat: StagyyData) -> None:
    series = sdat.tseries["Tmean"]
    assert series.values.shape == series.time.shape
    assert series.meta == stagpy.phyvars.TIME["Tmean"]


def test_get_time_series_extra(sdat: StagyyData) -> None:
    series = sdat.tseries["dTdt"]
    assert series.values.shape == series.time.shape == (sdat.tseries.time.size - 1,)


def test_compstat(sdat: StagyyData) -> None:
    variables = ["Tmean", "vrms"]
    stats = stagpy.time_series.compstat(sdat, *variables)
    assert list(stats.columns) == variables
    assert list(stats.index) == ["mean", "rms"]
