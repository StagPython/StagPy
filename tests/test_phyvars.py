from itertools import chain

from stagpy import phyvars


def test_dim() -> None:
    allvars = chain(
        phyvars.FIELD.values(), phyvars.RPROF.values(), phyvars.TIME.values()
    )
    for var in allvars:
        if var.dim != "1":  # type: ignore
            assert var.dim in phyvars.SCALES  # type: ignore
