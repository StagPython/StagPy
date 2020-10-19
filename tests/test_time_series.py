import stagpy.phyvars
import stagpy.time_series


def test_get_time_series(sdat):
    series, time, meta = sdat.tseries['Tmean']
    assert series.shape == time.shape
    assert meta == stagpy.phyvars.TIME['Tmean']


def test_get_time_series_extra(sdat):
    series, time, meta = sdat.tseries['dTdt']
    assert series.shape == time.shape == (sdat.tseries.time.size - 1,)
    assert isinstance(meta, stagpy.phyvars.Vart)


def test_compstat(sdat):
    variables = ['Tmean', 'vrms']
    stats = stagpy.time_series.compstat(sdat, *variables)
    assert list(stats.columns) == variables
    assert list(stats.index) == ['mean', 'rms']
