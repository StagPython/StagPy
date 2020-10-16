import stagpy.rprof
import stagpy.phyvars


def test_get_rprof(step):
    prof, rad, meta = step.rprofs['Tmean']
    assert rad is step.rprofs.centers
    assert prof.shape == (step.geom.nztot,)
    assert meta == stagpy.phyvars.RPROF['Tmean']


def test_get_rprof_extra(step):
    prof, rad, meta = step.rprofs['diff']
    assert rad is step.rprofs.walls
    assert prof.shape == rad.shape
    assert isinstance(meta, stagpy.phyvars.Varr)
