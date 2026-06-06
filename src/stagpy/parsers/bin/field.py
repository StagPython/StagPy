"""Parser of legacy field output."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from itertools import product

import numpy as np

from ...error import ParsingError
from ._cursor import Cursor

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, BinaryIO

    from numpy.typing import NDArray


@dataclass(frozen=True)
class _HeaderInfo:
    """Header information."""

    magic: int
    nval: int
    sfield: bool
    cursor: Cursor
    header: dict[str, Any]


def _header(filepath: Path, fid: BinaryIO, stop_at_istep: bool = False) -> _HeaderInfo:
    """Read the header of a legacy binary file."""
    cursor = Cursor(fid=fid, int_type=np.int32, float_type=np.float32)
    magic = cursor.single_int().item()
    if magic > 8000:  # 64 bits
        cursor = cursor.reset_with_64_bits()
        if magic != cursor.single_int():
            raise ParsingError(filepath, "inconsistent magic number in 64 bits")
        magic -= 8000

    # check nb components
    nval = 1
    sfield = False
    if magic > 400:
        nval = 4
    elif magic > 300:
        nval = 3
    elif magic > 100:
        sfield = True

    magic %= 100
    if magic < 9 or magic > 12:
        raise ParsingError(filepath, f"{magic=:d} not supported")

    header_info = _HeaderInfo(magic, nval, sfield, cursor, {})
    header = header_info.header
    # extra ghost point in horizontal direction
    header["xyp"] = int(nval == 4)  # magic >= 9

    # total number of values in relevant space basis
    # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
    #              = (x, y, z)            in cartesian geometry
    header["nts"] = cursor.ints(3)

    # number of blocks, 2 for yinyang or cubed sphere
    header["ntb"] = cursor.single_int()  # magic >= 7

    # aspect ratio
    header["aspect"] = cursor.floats(2)

    # number of parallel subdomains
    header["ncs"] = cursor.ints(3)  # (e1, e2, e3) space
    header["ncb"] = cursor.single_int()  # magic >= 8, blocks

    # r - coordinates
    # rgeom[0:self.nrtot+1, 0] are edge radial position
    # rgeom[0:self.nrtot, 1] are cell-center radial position
    header["rgeom"] = cursor.floats(header["nts"][2] * 2 + 1)  # magic >= 2
    header["rgeom"] = np.resize(header["rgeom"], (header["nts"][2] + 1, 2))

    header["rcmb"] = cursor.single_float()  # magic >= 7

    header["ti_step"] = cursor.single_int()  # magic >= 3
    if stop_at_istep:
        return header_info

    header["ti_ad"] = cursor.single_float()  # magic >= 3
    header["erupta_total"] = cursor.single_float()  # magic >= 5
    if magic >= 12:
        header["erupta_ttg"] = cursor.single_float()
        header["intruda"] = cursor.floats(2)
        header["ttg_mass"] = cursor.floats(3)
    else:
        header["erupta_ttg"] = 0.0
        header["intruda"] = np.zeros(2)
        header["ttg_mass"] = np.zeros(3)
    header["bot_temp"] = cursor.single_float()  # magic >= 6
    header["core_temp"] = cursor.single_float() if magic >= 10 else 1.0
    header["ocean_mass"] = cursor.single_float() if magic >= 11 else 0.0

    # magic >= 4
    header["e1_coord"] = cursor.floats(header["nts"][0])
    header["e2_coord"] = cursor.floats(header["nts"][1])
    header["e3_coord"] = cursor.floats(header["nts"][2])

    return header_info


def istep(fieldfile: Path) -> int | None:
    """Read istep from binary field file.

    Args:
        fieldfile: path of the binary field file.

    Returns:
        the time step at which the binary file was written.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = _header(fieldfile, fid, stop_at_istep=True)
    return hdr.header["ti_step"]


def header(fieldfile: Path) -> dict[str, Any] | None:
    """Read header info from binary field file.

    Args:
        fieldfile: path of the binary field file.

    Returns:
        the header information of the binary file.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = _header(fieldfile, fid)
    return hdr.header


def field(fieldfile: Path) -> tuple[dict[str, Any], NDArray[np.float64]] | None:
    """Extract fields data.

    Args:
        fieldfile: path of the binary field file.

    Returns:
        the tuple `(header, fields)`. `fields` is an array of scalar fields
            indexed by variable, x-direction, y-direction, z-direction, block.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = _header(fieldfile, fid)
        header = hdr.header
        cursor = hdr.cursor

        # number of points in (e1, e2, e3) directions per cpu
        npc = header["nts"] // header["ncs"]
        # number of blocks per cpu
        nbk = header["ntb"] // header["ncb"]
        # number of values per 'read' block
        npi = (
            (npc[0] + header["xyp"])
            * (npc[1] + header["xyp"])
            * npc[2]
            * nbk
            * hdr.nval
        )

        header["scalefac"] = cursor.single_float() if hdr.nval > 1 else 1.0

        flds = np.zeros(
            (
                hdr.nval,
                header["nts"][0] + header["xyp"],
                header["nts"][1] + header["xyp"],
                header["nts"][2],
                header["ntb"],
            )
        )

        # loop over parallel subdomains
        for icpu in product(
            range(header["ncb"]),
            range(header["ncs"][2]),
            range(header["ncs"][1]),
            range(header["ncs"][0]),
        ):
            # read the data for one CPU
            data_cpu = cursor.floats(npi) * header["scalefac"]

            # icpu is (icpu block, icpu z, icpu y, icpu x)
            # data from file is transposed to obtained a field
            # array indexed with (x, y, z, block), as in StagYY
            flds[
                :,
                icpu[3] * npc[0] : (icpu[3] + 1) * npc[0] + header["xyp"],  # x
                icpu[2] * npc[1] : (icpu[2] + 1) * npc[1] + header["xyp"],  # y
                icpu[1] * npc[2] : (icpu[1] + 1) * npc[2],  # z
                icpu[0] * nbk : (icpu[0] + 1) * nbk,  # block
            ] = np.transpose(
                data_cpu.reshape(
                    (
                        nbk,
                        npc[2],
                        npc[1] + header["xyp"],
                        npc[0] + header["xyp"],
                        hdr.nval,
                    )
                )
            )
        if hdr.sfield:
            # for surface fields, variables are written along z direction
            flds = np.swapaxes(flds, 0, 3)
    return header, flds
