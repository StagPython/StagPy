from __future__ import annotations

import typing
from operator import itemgetter

import h5py
import pandas as pd

from ..._helpers import resize

if typing.TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


def rprof(rproffile: Path) -> tuple[dict[int, DataFrame], DataFrame | None]:
    """Extract radial profiles data.

    Args:
        rproffile: path of the rprof.h5 file.

    Returns:
        profs: a dict mapping istep to radial profiles.
        times: the time indexed by time steps.
    """
    if not rproffile.is_file():
        return {}, None
    isteps = []
    data = {}
    with h5py.File(rproffile, "r") as h5f:
        dnames = sorted(dname for dname in h5f.keys() if dname.startswith("rprof_"))
        colnames = h5f["names"].asstr()[()]
        for dname in dnames:
            dset = h5f[dname]
            arr = dset[()]
            istep = dset.attrs["istep"]
            step_cols = list(colnames)
            resize(step_cols, arr.shape[1])  # check shape
            data[istep] = pd.DataFrame(arr, columns=step_cols)
            isteps.append((istep, dset.attrs["time"]))

    df_times = pd.DataFrame(
        list(map(itemgetter(1), isteps)), index=pd.Index(map(itemgetter(0), isteps))
    )
    return data, df_times
