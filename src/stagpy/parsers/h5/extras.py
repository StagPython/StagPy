from __future__ import annotations

import typing

import h5py

if typing.TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def isnap_istep(h5folder: Path) -> Iterator[tuple[int, int]]:
    """Iterate through (isnap, istep) recorded in h5folder/'time_botT.h5'.

    Args:
        h5folder: directory of HDF5 output files.

    Yields:
        tuple (isnap, istep).
    """
    with h5py.File(h5folder / "time_botT.h5", "r") as h5f:
        for name, dset in h5f.items():
            isnap = int(name[-5:])
            if len(dset) == 3:
                istep = int(dset[2])
            else:
                istep = int(dset[0])
            yield isnap, istep
