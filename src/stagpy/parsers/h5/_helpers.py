from __future__ import annotations

import typing

import h5py
import numpy as np

from ...error import ParsingError

if typing.TYPE_CHECKING:
    from pathlib import Path
    from xml.etree.ElementTree import Element

    from numpy.typing import NDArray

    from .xdmf import XmlStream


def read_group(filename: Path, groupname: str) -> NDArray[np.float64]:
    """Return group content.

    Args:
        filename: path of hdf5 file.
        groupname: name of group to read.

    Returns:
        content of group.
    """
    try:
        with h5py.File(filename, "r") as h5f:
            data = h5f[groupname][()]
    except OSError as err:
        # h5py doesn't always include the filename in its error messages
        err.args += (filename,)
        raise
    return data  # need to be reshaped


def try_text(file: Path, elt: Element) -> str:
    """Try getting text of element or raise a ParsingError."""
    text = elt.text
    if text is None:
        raise ParsingError(file, f"Element {elt} has no 'text'")
    return text


def ifile_isnap(file: Path, elt: Element) -> tuple[int, int]:
    """Extract ifile and isnap from H5 file name/dataset."""
    data_text = try_text(file, elt)
    h5file, group = data_text.strip().split(":/", 1)
    isnap = int(group[-5:])
    ifile = int(h5file[-14:-9])
    return ifile, isnap


def count_subdomains(xs: XmlStream, i0_yin: int) -> tuple[range, range]:
    i1_yin = i0_yin + 1
    i0_yang = 0
    i1_yang = 0
    for _ in xs.iter_tag("Grid"):
        if xs.current.attrib["GridType"] == "Collection":
            break
        if (name := xs.current.attrib["Name"]).startswith("meshYang"):
            if i1_yang == 0:
                i0_yang = int(name[-5:]) - 1
                i1_yang = i0_yang + (i1_yin - i0_yin)
        else:
            i1_yin += 1
        xs.drop()
    return range(i0_yin, i1_yin), range(i0_yang, i1_yang)
