from __future__ import annotations

import typing

import numpy as np

from ...error import ParsingError
from ._cursor import Cursor

if typing.TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def tracers(tracersfile: Path) -> dict[str, list[NDArray[np.floating]]] | None:
    """Extract tracers data.

    Args:
        tracersfile: path of the binary tracers file.

    Returns:
        Tracers data organized by attribute names and blocks.
    """
    if not tracersfile.is_file():
        return None
    tra: dict[str, list[NDArray[np.floating]]] = {}
    with tracersfile.open("rb") as fid:
        cursor = Cursor(fid=fid, int_type=np.int32, float_type=np.float32)
        magic = cursor.single_int().item()
        if magic > 8000:  # 64 bits
            cursor = cursor.reset_with_64_bits()
            if magic != cursor.single_int():
                raise ParsingError(tracersfile, "inconsistent magic number in 64 bits")
            magic -= 8000
        if magic < 100:
            raise ParsingError(
                tracersfile, "magic > 100 expected to get tracervar info"
            )
        nblk = magic % 100
        cursor.floats(2)  # aspect ratio
        cursor.single_int()  # istep
        cursor.single_float()  # time
        ninfo = cursor.single_int()
        ntra = cursor.ints(nblk)
        cursor.single_float()  # tracer ideal mass
        curv = cursor.single_int()
        if curv:
            cursor.single_float()  # r_cmb
        infos = []  # list of info names
        for _ in range(ninfo):
            infos.append(cursor.string(16))
            tra[infos[-1]] = []
        if magic > 200:
            ntrace_elt = cursor.single_int()
            if ntrace_elt > 0:
                cursor.floats(ntrace_elt)  # outgassed
        for ntrab in ntra:  # blocks
            data = cursor.floats(ntrab * ninfo)
            for idx, info in enumerate(infos):
                tra[info].append(data[idx::ninfo])
    return tra
