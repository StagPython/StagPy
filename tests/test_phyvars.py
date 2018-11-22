from itertools import chain
from stagpy import phyvars


def test_dim():
    allvars = chain(phyvars.FIELD.values(), phyvars.FIELD_EXTRA.values(),
                    phyvars.RPROF.values(), phyvars.RPROF_EXTRA.values(),
                    phyvars.TIME.values(), phyvars.TIME_EXTRA.values())
    for var in allvars:
        if var.dim != '1':
            assert var.dim in phyvars.SCALES
