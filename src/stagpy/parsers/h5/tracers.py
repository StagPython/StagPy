from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ...error import ParsingError
from ._helpers import count_subdomains, ifile_isnap, read_group
from .xdmf import XmlStream

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from numpy.typing import NDArray


@dataclass(frozen=True)
class TracerSub:
    file: Path
    dataset: str
    icore: int
    iblock: int


@dataclass(frozen=True)
class XmfTracersEntry:
    isnap: int
    time: float | None
    mo_lambda: float | None
    mo_thick_sol: float | None
    yin_yang: bool
    range_yin: range
    range_yang: range
    fields: Mapping[str, int]

    def _fsub(self, path_root: Path, name: str, icore: int, yang: bool) -> TracerSub:
        ifile = self.fields[name]
        yy_tag = "2" if yang else "_"
        ic = icore + 1
        return TracerSub(
            file=path_root / f"Tracers_{name}_{ifile:05d}_{ic:05d}.h5",
            dataset=f"{name}{yy_tag}{ic:05d}_{self.isnap:05d}",
            icore=icore,
            iblock=int(yang),
        )

    def tra_subdomains(self, path_root: Path, name: str) -> Iterator[TracerSub]:
        if name not in self.fields:
            return
        for icore in self.range_yin:
            yield self._fsub(path_root, name, icore, False)
        for icore in self.range_yang:
            yield self._fsub(path_root, name, icore, True)


@dataclass(frozen=True)
class TracersXmf:
    path: Path

    @cached_property
    def _data(self) -> Mapping[int, XmfTracersEntry]:
        xs = XmlStream(filepath=self.path)
        data = {}
        for _ in xs.iter_tag("Time"):
            time = float(xs.current.attrib["Value"])
            xs.advance()
            extra: dict[str, float] = {}
            while xs.current.tag != "Grid":
                # mo_lambda, mo_thick_sol
                extra[xs.current.tag] = float(xs.current.attrib["Value"])
                xs.advance()

            mesh_name = xs.current.attrib["Name"]
            yin_yang = mesh_name.startswith("meshYin")
            i0_yin = int(mesh_name[-5:]) - 1

            fields_info = {}

            xs.skip_to_tag("Geometry")
            with xs.load() as elt_geom:
                for name, data_item in zip("zyx", elt_geom):
                    ifile, isnap = ifile_isnap(xs.filepath, data_item)
                    fields_info[name] = ifile

            for elt_fvar in xs.iter_load_successive_tag("Attribute"):
                name = elt_fvar.attrib["Name"]
                ifile, _ = ifile_isnap(xs.filepath, elt_fvar[0])
                fields_info[name] = ifile

            r_yin, r_yang = count_subdomains(xs, i0_yin)

            data[isnap] = XmfTracersEntry(
                isnap=isnap,
                time=time,
                mo_lambda=extra.get("mo_lambda"),
                mo_thick_sol=extra.get("mo_thick_sol"),
                yin_yang=yin_yang,
                range_yin=r_yin,
                range_yang=r_yang,
                fields=fields_info,
            )
        return data

    def __getitem__(self, isnap: int) -> XmfTracersEntry:
        try:
            return self._data[isnap]
        except KeyError:
            raise ParsingError(self.path, f"no data for snapshot {isnap}")


def tracers(
    xdmf: TracersXmf, infoname: str, snapshot: int
) -> list[NDArray[np.float64]]:
    """Extract tracers data from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        infoname: name of information to extract.
        snapshot: snapshot number.

    Returns:
        Tracers data organized by attribute and block.
    """
    tra: list[list[NDArray[np.float64]]] = [[], []]  # [block][core]
    for tsub in xdmf[snapshot].tra_subdomains(xdmf.path.parent, infoname):
        tra[tsub.iblock].append(read_group(tsub.file, tsub.dataset))

    tra_concat: list[NDArray[np.float64]] = []
    for trab in tra:
        if trab:
            tra_concat.append(np.concatenate(trab))
    return tra_concat
