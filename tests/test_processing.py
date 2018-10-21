from stagpy import processing


def test_dt_dt(sdat):
    dtdt, time = processing.dt_dt(sdat)
    assert dtdt.shape == time.shape == (sdat.tseries.shape[0] - 1,)


def test_ebalance(sdat):
    ebal, time = processing.ebalance(sdat)
    assert ebal.shape == time.shape == (sdat.tseries.shape[0] - 1,)


def test_r_edges(step):
    edges1, edges2 = processing.r_edges(step)
    assert edges1 is edges2
    assert edges1.shape == (step.rprof.shape[0] + 1,)


def test_delta_r(step):
    thick, _ = processing.delta_r(step)
    assert thick.shape == (step.rprof.shape[0],)


def test_diff_prof(step):
    diff, rpos = processing.diff_prof(step)
    assert diff.shape == rpos.shape == (step.rprof.shape[0] + 1,)


def test_diffs_prof(step):
    diff, rpos = processing.diffs_prof(step)
    assert diff.shape == rpos.shape == (step.rprof.shape[0] + 1,)


def test_advts_prof(step):
    adv, _ = processing.advts_prof(step)
    assert adv.shape == (step.rprof.shape[0],)


def test_advds_prof(step):
    adv, _ = processing.advds_prof(step)
    assert adv.shape == (step.rprof.shape[0],)


def test_advas_prof(step):
    adv, _ = processing.advas_prof(step)
    assert adv.shape == (step.rprof.shape[0],)


def test_energy_prof(step):
    eng, rpos = processing.energy_prof(step)
    assert eng.shape == rpos.shape == (step.rprof.shape[0] + 1,)


def test_stream_function(step):
    psi = processing.stream_function(step)
    assert psi.shape[1:3] == step.fields['v3'].shape[1:3]
