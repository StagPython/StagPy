import stagpy.rprof
import stagpy.phyvars


def test_get_rprof(step):
    prof, rad, meta = stagpy.rprof.get_rprof(step, 'Tmean')
    assert rad is None
    assert prof.shape == (step.rprof.shape[0],)
    assert meta == stagpy.phyvars.RPROF['Tmean']


def test_get_rprof_extra(step):
    prof, rad, meta = stagpy.rprof.get_rprof(step, 'diff')
    assert prof.shape == rad.shape == (step.rprof.shape[0] + 1,)
    assert isinstance(meta, stagpy.phyvars.Varr)
