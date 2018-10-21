import stagpy.phyvars
import stagpy.time_series


def test_get_time_series(sdat):
    series, time, meta = stagpy.time_series.get_time_series(
        sdat, 'Tmean', None, None)
    assert time is None
    assert series.shape == (sdat.tseries.shape[0],)
    assert meta == stagpy.phyvars.TIME['Tmean']


def test_get_time_series_extra(sdat):
    series, time, meta = stagpy.time_series.get_time_series(
        sdat, 'dTdt', None, None)
    assert series.shape == time.shape == (sdat.tseries.shape[0] - 1,)
    assert isinstance(meta, stagpy.phyvars.Vart)
