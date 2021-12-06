from stagpy import processing, phyvars


def tseries_checks(tseries, expected_size):
    assert tseries.values.shape == tseries.time.shape == (expected_size,)
    assert tseries.meta.dim == '1' or tseries.meta.dim in phyvars.SCALES


def rprof_checks(rprof, expected_size):
    assert rprof.values.shape == rprof.rad.shape == (expected_size,)
    assert rprof.meta.dim == '1' or rprof.meta.dim in phyvars.SCALES


def test_dt_dt(sdat):
    tseries_checks(processing.dt_dt(sdat), sdat.tseries.time.shape[0] - 1)


def test_ebalance(sdat):
    tseries_checks(processing.ebalance(sdat), sdat.tseries.time.shape[0] - 1)


def test_r_edges(step):
    assert step.rprofs.walls.shape == (step.geom.nztot + 1,)


def test_delta_r(step):
    rprof_checks(processing.delta_r(step), step.geom.nztot)


def test_diff_prof(step):
    rprof_checks(processing.diff_prof(step), step.geom.nztot + 1)


def test_diffs_prof(step):
    rprof_checks(processing.diffs_prof(step), step.geom.nztot + 1)


def test_advts_prof(step):
    rprof_checks(processing.advts_prof(step), step.geom.nztot)


def test_advds_prof(step):
    rprof_checks(processing.advds_prof(step), step.geom.nztot)


def test_advas_prof(step):
    rprof_checks(processing.advas_prof(step), step.geom.nztot)


def test_energy_prof(step):
    rprof_checks(processing.energy_prof(step), step.geom.nztot + 1)


def test_stream_function(step):
    psi = processing.stream_function(step)
    assert psi.values.shape[1:3] == step.fields['v3'].values.shape[1:3]
    assert psi.meta.dim in phyvars.SCALES
