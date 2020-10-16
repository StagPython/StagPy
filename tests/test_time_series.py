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
