from __future__ import annotations

import typing

import h5py
import numpy as np
import pandas as pd

from ..._helpers import resize

if typing.TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


def tseries(timefile: Path) -> DataFrame | None:
    """Read temporal series HDF5 file.

    Args:
        timefile: path of the TimeSeries.h5 file.

    Returns:
        A `pandas.DataFrame` containing the time series, organized by
            variables in columns and the time steps in rows.
    """
    if not timefile.is_file():
        return None
    with h5py.File(timefile, "r") as h5f:
        dset = h5f["tseries"]
        _, ncols = dset.shape
        ncols -= 1  # first is istep
        colnames = list(h5f["names"].asstr()[()])
        resize(colnames, ncols + 1)
        data = dset[()]
    pdf = pd.DataFrame(
        data[:, 1:],
        index=data[:, 0].astype(np.int64),
        columns=colnames[1:],
    )
    # remove duplicated lines in case of restart
    return pdf.loc[~pdf.index.duplicated(keep="last")]
