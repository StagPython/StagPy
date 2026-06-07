from __future__ import annotations

import typing

import h5py

if typing.TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def isnap_istep(timeh5: Path) -> Iterator[tuple[int, int]]:
    """Iterate through (isnap, istep) recorded in 'time_botT.h5'.

    Args:
        timeh5: path of the time h5 file.

    Yields:
        tuple (isnap, istep).
    """
    with h5py.File(timeh5, "r") as h5f:
        for name, dset in h5f.items():
            isnap = int(name[-5:])
            if len(dset) == 3:
                istep = int(dset[2])
            else:
                istep = int(dset[0])
            yield isnap, istep
