from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    from typing import BinaryIO

    from numpy.typing import NDArray


@dataclass(frozen=True)
class Cursor:
    fid: BinaryIO
    int_type: type[np.integer]
    float_type: type[np.floating]

    def reset_with_64_bits(self) -> Cursor:
        self.fid.seek(0)
        return Cursor(
            fid=self.fid,
            int_type=np.int64,
            float_type=np.float64,
        )

    def string(self, nbytes: int) -> str:
        return b"".join(np.fromfile(self.fid, "b", nbytes)).strip().decode()

    def single_int(self) -> np.integer:
        return np.fromfile(self.fid, self.int_type, 1)[0]

    def single_float(self) -> np.floating:
        return np.fromfile(self.fid, self.float_type, 1)[0]

    def ints(self, count: int | np.integer) -> NDArray[np.integer]:
        return np.fromfile(self.fid, self.int_type, count)

    def floats(self, count: int | np.integer) -> NDArray[np.floating]:
        return np.fromfile(self.fid, self.float_type, count)
