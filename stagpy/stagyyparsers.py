"""Parsers of StagYY output files.

Note:
    These functions are low level utilities. You should not use these unless
    you know what you are doing. To access StagYY output data, use an instance
    of [`StagyyData`][stagpy.stagyydata.StagyyData].
"""

from __future__ import annotations

import re
import typing
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import product
from operator import itemgetter

import h5py
import numpy as np
import pandas as pd

from .error import ParsingError
from .phyvars import FIELD_FILES_H5, SFIELD_FILES_H5
from .xdmf import XmlStream

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, BinaryIO, Callable, Iterator, Mapping, Optional
    from xml.etree.ElementTree import Element

    from numpy.typing import NDArray
    from pandas import DataFrame


def _tidy_names(
    names: list[str], nnames: int, extra_names: Optional[list[str]] = None
) -> None:
    """Truncate or extend names so that its len is nnames.

    The list is modified in-place.

    Args:
        names: list of names.
        nnames: desired number of names.
        extra_names: list of names to be used to extend the list if needed. If
            this list isn't provided, a range is used instead.
    """
    if len(names) < nnames and extra_names is not None:
        names.extend(extra_names)
    names.extend(map(str, range(nnames - len(names))))
    del names[nnames:]


def time_series(timefile: Path, colnames: list[str]) -> Optional[DataFrame]:
    """Read temporal series text file.

    If `colnames` is too long, it will be truncated. If it is too short,
    additional numeric column names from 0 to N-1 will be attributed to the N
    extra columns present in `timefile`.

    Args:
        timefile: path of the time.dat file.
        colnames: names of the variables expected in `timefile` (may be modified).

    Returns:
        A `pandas.DataFrame` containing the time series, organized by
            variables in columns and time steps in rows.
    """
    if not timefile.is_file():
        return None
    data = pd.read_csv(
        timefile,
        sep=r"\s+",
        dtype=str,
        header=None,
        skiprows=1,
        index_col=0,
        engine="c",
        memory_map=True,
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")  # type: ignore

    # detect useless lines produced when run is restarted
    rows_to_del = []
    irow = len(data) - 1
    while irow > 0:
        iprev = irow - 1
        while iprev >= 0 and data.index[irow] <= data.index[iprev]:
            rows_to_del.append(iprev)
            iprev -= 1
        irow = iprev
    if rows_to_del:
        rows_to_keep = set(range(len(data))) - set(rows_to_del)
        data = data.take(list(rows_to_keep))

    ncols = data.shape[1]
    _tidy_names(colnames, ncols)
    data.columns = pd.Index(colnames)

    return data


def time_series_h5(timefile: Path, colnames: list[str]) -> Optional[DataFrame]:
    """Read temporal series HDF5 file.

    If `colnames` is too long, it will be truncated. If it is too short,
    additional column names will be deduced from the content of the file.

    Args:
        timefile: path of the TimeSeries.h5 file.
        colnames: names of the variables expected in `timefile` (may be modified).

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
        h5names = h5f["names"].asstr()[len(colnames) + 1 :]
        _tidy_names(colnames, ncols, h5names)
        data = dset[()]
    pdf = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(np.int64), columns=colnames)
    # remove duplicated lines in case of restart
    return pdf.loc[~pdf.index.duplicated(keep="last")]


def _extract_rsnap_isteps(
    rproffile: Path, data: DataFrame
) -> list[tuple[int, float, DataFrame]]:
    """Extract istep, time and build separate rprof df."""
    step_regex = re.compile(r"^\*+step:\s*(\d+) ; time =\s*(\S+)")
    isteps = []  # list of (istep, time, df)
    line = " "
    with rproffile.open() as stream:
        while line[0] != "*":
            line = stream.readline()
        match = step_regex.match(line)
        if match is None:
            raise ParsingError(rproffile, f"Badly formatted line {line!r}")
        istep = int(match.group(1))
        time = float(match.group(2))
        nlines = 0
        iline = 0
        for line in stream:
            if line[0] == "*":
                isteps.append((istep, time, data.iloc[iline - nlines : iline]))
                match = step_regex.match(line)
                if match is None:
                    raise ParsingError(rproffile, f"Badly formatted line {line!r}")
                istep = int(match.group(1))
                time = float(match.group(2))
                nlines = 0
                # remove useless lines produced when run is restarted
                while isteps and istep <= isteps[-1][0]:
                    isteps.pop()
            else:
                nlines += 1
                iline += 1
        isteps.append((istep, time, data.iloc[iline - nlines : iline]))
    return isteps


def rprof(
    rproffile: Path, colnames: list[str]
) -> tuple[dict[int, DataFrame], Optional[DataFrame]]:
    """Extract radial profiles data.

    If `colnames` is too long, it will be truncated. If it is too short,
    additional numeric column names from 0 to N-1 will be attributed to the N
    extra columns present in `timefile`.

    Args:
        rproffile: path of the rprof.dat file.
        colnames: names of the variables expected in `rproffile`.

    Returns:
        profs: a dict mapping istep to radial profiles.
        times: the time indexed by time steps.
    """
    if not rproffile.is_file():
        return {}, None
    data = pd.read_csv(
        rproffile,
        sep=r"\s+",
        dtype=str,
        header=None,
        comment="*",
        skiprows=1,
        engine="c",
        memory_map=True,
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")  # type: ignore

    isteps = _extract_rsnap_isteps(rproffile, data)

    all_data = {}
    for istep, _, step_df in isteps:
        step_df.index = pd.RangeIndex(step_df.shape[0])  # check whether necessary
        step_cols = list(colnames)
        _tidy_names(step_cols, step_df.shape[1])
        step_df.columns = pd.Index(step_cols)
        all_data[istep] = step_df

    df_times = pd.DataFrame(
        list(map(itemgetter(1), isteps)), index=pd.Index(map(itemgetter(0), isteps))
    )
    return all_data, df_times


def rprof_h5(
    rproffile: Path, colnames: list[str]
) -> tuple[dict[int, DataFrame], Optional[DataFrame]]:
    """Extract radial profiles data.

    If `colnames` is too long, it will be truncated. If it is too short,
    additional column names will be deduced from the content of the file.

    Args:
        rproffile: path of the rprof.h5 file.
        colnames: names of the variables expected in `rproffile`.

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
        h5names = h5f["names"].asstr()[len(colnames) :]
        for dname in dnames:
            dset = h5f[dname]
            arr = dset[()]
            istep = dset.attrs["istep"]
            step_cols = list(colnames)
            _tidy_names(step_cols, arr.shape[1], h5names)  # check shape
            data[istep] = pd.DataFrame(arr, columns=step_cols)
            isteps.append((istep, dset.attrs["time"]))

    df_times = pd.DataFrame(
        list(map(itemgetter(1), isteps)), index=pd.Index(map(itemgetter(0), isteps))
    )
    return data, df_times


def _clean_names_refstate(names: list[str]) -> list[str]:
    """Uniformization of refstate profile names."""
    to_clean = {
        "Tref": "T",
        "rhoref": "rho",
        "tcond": "Tcond",
    }
    return [to_clean.get(n, n) for n in names]


def refstate(
    reffile: Path, ncols: int = 8
) -> Optional[tuple[list[list[DataFrame]], list[DataFrame]]]:
    """Extract reference state profiles.

    Args:
        reffile: path of the refstate file.
        ncols: number of columns.

    Returns:
        syst: list of list of `pandas.DataFrame` containing the reference
            state profiles for each system and each phase in these systems.
        adia: list of `pandas.DataFrame` containing the adiabatic reference
            state profiles for each system, the last item being the combined
            adiabat.
    """
    if not reffile.is_file():
        return None
    data = pd.read_csv(
        reffile,
        sep=r"\s+",
        dtype=str,
        header=None,
        names=range(ncols),
        engine="c",
        memory_map=True,
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")  # type: ignore
    # drop lines corresponding to metadata
    data.dropna(subset=[0], inplace=True)
    isystem = -1
    systems: list[list[list[str]]] = [[]]
    adiabats: list[list[str]] = []
    with reffile.open() as rsf:
        for line in rsf:
            line = line.lstrip()
            if line.startswith("SYSTEM"):
                isystem += 1
                if isystem > 0:
                    systems.append([])
            elif line.startswith("z"):
                systems[isystem].append(_clean_names_refstate(line.split()))
            elif line.startswith("ADIABAT") or line.startswith("COMBINED"):
                line = line.partition(":")[-1]
                adiabats.append(_clean_names_refstate(line.split()))
    nprofs = sum(map(len, systems)) + len(adiabats)
    nzprof = len(data) // nprofs
    iprof = 0
    syst: list[list[DataFrame]] = []
    adia: list[DataFrame] = []
    for isys, layers in enumerate(systems):
        syst.append([])
        for layer in layers:
            ibgn = iprof * nzprof
            iend = ibgn + nzprof
            syst[isys].append(
                pd.DataFrame(data.iloc[ibgn:iend, : len(layer)].values, columns=layer)
            )
            iprof += 1
        if len(layers) > 1:
            ibgn = iprof * nzprof
            iend = ibgn + nzprof
            cols = adiabats.pop(0)
            adia.append(
                pd.DataFrame(data.iloc[ibgn:iend, : len(cols)].values, columns=cols)
            )
            iprof += 1
        else:
            adia.append(syst[isys][0])
    ibgn = iprof * nzprof
    iend = ibgn + nzprof
    cols = adiabats.pop(0)
    adia.append(pd.DataFrame(data.iloc[ibgn:iend, : len(cols)].values, columns=cols))
    return syst, adia


def _readbin(
    fid: BinaryIO,
    fmt: str = "i",
    nwords: int = 1,
    file64: bool = False,
    unpack: bool = True,
) -> Any:
    """Read n words of 4 or 8 bytes with fmt format.

    fmt: 'i' or 'f' or 'b' (integer or float or bytes)
    4 or 8 bytes: depends on header

    Return an array of elements if more than one element.

    Default: read 1 word formatted as an integer.
    """
    if fmt in "if":
        fmt += "8" if file64 else "4"
    elts = np.fromfile(fid, fmt, nwords)
    if unpack and len(elts) == 1:
        elts = elts[0]
    return elts


@dataclass(frozen=True)
class _HeaderInfo:
    """Header information."""

    magic: int
    nval: int
    sfield: bool
    readbin: Callable
    header: dict[str, Any]


def _legacy_header(
    filepath: Path, fid: BinaryIO, stop_at_istep: bool = False
) -> _HeaderInfo:
    """Read the header of a legacy binary file."""
    readbin = partial(_readbin, fid)
    magic = readbin()
    if magic > 8000:  # 64 bits
        magic -= 8000
        readbin()  # need to read 4 more bytes
        readbin = partial(readbin, file64=True)

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
    if magic < 9 or magic > 11:
        raise ParsingError(filepath, f"{magic=} not supported")

    header_info = _HeaderInfo(magic, nval, sfield, readbin, {})
    header = header_info.header
    # extra ghost point in horizontal direction
    header["xyp"] = int(nval == 4)  # magic >= 9

    # total number of values in relevant space basis
    # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
    #              = (x, y, z)            in cartesian geometry
    header["nts"] = readbin(nwords=3)

    # number of blocks, 2 for yinyang or cubed sphere
    header["ntb"] = readbin()  # magic >= 7

    # aspect ratio
    header["aspect"] = readbin("f", 2)

    # number of parallel subdomains
    header["ncs"] = readbin(nwords=3)  # (e1, e2, e3) space
    header["ncb"] = readbin()  # magic >= 8, blocks

    # r - coordinates
    # rgeom[0:self.nrtot+1, 0] are edge radial position
    # rgeom[0:self.nrtot, 1] are cell-center radial position
    header["rgeom"] = readbin("f", header["nts"][2] * 2 + 1)  # magic >= 2
    header["rgeom"] = np.resize(header["rgeom"], (header["nts"][2] + 1, 2))

    header["rcmb"] = readbin("f")  # magic >= 7

    header["ti_step"] = readbin()  # magic >= 3
    if stop_at_istep:
        return header_info

    header["ti_ad"] = readbin("f")  # magic >= 3
    header["erupta_total"] = readbin("f")  # magic >= 5
    header["bot_temp"] = readbin("f")  # magic >= 6
    header["core_temp"] = readbin("f") if magic >= 10 else 1
    header["ocean_mass"] = readbin("f") if magic >= 11 else 0.0

    # magic >= 4
    header["e1_coord"] = readbin("f", header["nts"][0])
    header["e2_coord"] = readbin("f", header["nts"][1])
    header["e3_coord"] = readbin("f", header["nts"][2])

    return header_info


def field_istep(fieldfile: Path) -> Optional[int]:
    """Read istep from binary field file.

    Args:
        fieldfile: path of the binary field file.

    Returns:
        the time step at which the binary file was written.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = _legacy_header(fieldfile, fid, stop_at_istep=True)
    return hdr.header["ti_step"]


def field_header(fieldfile: Path) -> Optional[dict[str, Any]]:
    """Read header info from binary field file.

    Args:
        fieldfile: path of the binary field file.

    Returns:
        the header information of the binary file.
    """
    if not fieldfile.is_file():
        return None
    with fieldfile.open("rb") as fid:
        hdr = _legacy_header(fieldfile, fid)
    return hdr.header


def fields(fieldfile: Path) -> Optional[tuple[dict[str, Any], NDArray]]:
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
        hdr = _legacy_header(fieldfile, fid)
        header = hdr.header

        # READ FIELDS
        # number of points in (e1, e2, e3) directions PER CPU
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

        header["scalefac"] = hdr.readbin("f") if hdr.nval > 1 else 1

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
            data_cpu = hdr.readbin("f", npi) * header["scalefac"]

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


def tracers(tracersfile: Path) -> Optional[dict[str, list[NDArray]]]:
    """Extract tracers data.

    Args:
        tracersfile: path of the binary tracers file.

    Returns:
        Tracers data organized by attribute names and blocks.
    """
    if not tracersfile.is_file():
        return None
    tra: dict[str, list[NDArray]] = {}
    with tracersfile.open("rb") as fid:
        readbin = partial(_readbin, fid)
        magic = readbin()
        if magic > 8000:  # 64 bits
            magic -= 8000
            readbin()
            readbin = partial(readbin, file64=True)
        if magic < 100:
            raise ParsingError(
                tracersfile, "magic > 100 expected to get tracervar info"
            )
        nblk = magic % 100
        readbin("f", 2)  # aspect ratio
        readbin()  # istep
        readbin("f")  # time
        ninfo = readbin()
        ntra = readbin(nwords=nblk, unpack=False)
        readbin("f")  # tracer ideal mass
        curv = readbin()
        if curv:
            readbin("f")  # r_cmb
        infos = []  # list of info names
        for _ in range(ninfo):
            infos.append(b"".join(readbin("b", 16)).strip().decode())
            tra[infos[-1]] = []
        if magic > 200:
            ntrace_elt = readbin()
            if ntrace_elt > 0:
                readbin("f", ntrace_elt)  # outgassed
        for ntrab in ntra:  # blocks
            data = readbin("f", ntrab * ninfo)
            for idx, info in enumerate(infos):
                tra[info].append(data[idx::ninfo])
    return tra


def _read_group_h5(filename: Path, groupname: str) -> NDArray:
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


def _make_3d(field: NDArray, twod: Optional[str]) -> NDArray:
    """Add a dimension to field if necessary.

    Args:
        field: the field that needs to be 3d.
        twod: 'XZ', 'YZ' or None depending on what is relevant.

    Returns:
        reshaped field.
    """
    shp = list(field.shape)
    if twod and "X" in twod:
        shp.insert(1, 1)
    elif twod:
        shp.insert(0, 1)
    return field.reshape(shp)


def _ncores(meshes: list[dict[str, NDArray]], twod: Optional[str]) -> NDArray:
    """Compute number of nodes in each direction."""
    nnpb = len(meshes)  # number of nodes per block
    nns = [1, 1, 1]  # number of nodes in x, y, z directions
    if twod is None or "X" in twod:
        while (
            nnpb > 1
            and meshes[nns[0]]["X"][0, 0, 0] == meshes[nns[0] - 1]["X"][-1, 0, 0]
        ):
            nns[0] += 1
            nnpb -= 1
    cpu = lambda icy: icy * nns[0]  # noqa: E731
    if twod is None or "Y" in twod:
        while (
            nnpb > 1
            and meshes[cpu(nns[1])]["Y"][0, 0, 0]
            == meshes[cpu(nns[1] - 1)]["Y"][0, -1, 0]
        ):
            nns[1] += 1
            nnpb -= nns[0]
    cpu = lambda icz: icz * nns[0] * nns[1]  # noqa: E731
    while (
        nnpb > 1
        and meshes[cpu(nns[2])]["Z"][0, 0, 0] == meshes[cpu(nns[2] - 1)]["Z"][0, 0, -1]
    ):
        nns[2] += 1
        nnpb -= nns[0] * nns[1]
    return np.array(nns)


def _conglomerate_meshes(
    meshin: list[dict[str, NDArray]], header: dict[str, Any]
) -> dict[str, NDArray]:
    """Conglomerate meshes from several cores into one."""
    meshout = {}
    npc = header["nts"] // header["ncs"]
    shp = [val + 1 if val != 1 else 1 for val in header["nts"]]
    x_p = int(shp[0] != 1)
    y_p = int(shp[1] != 1)
    for coord in meshin[0]:
        meshout[coord] = np.zeros(shp)
    for icore in range(np.prod(header["ncs"])):
        ifs = [
            icore // np.prod(header["ncs"][:i]) % header["ncs"][i] * npc[i]
            for i in range(3)
        ]
        for coord, mesh in meshin[icore].items():
            meshout[coord][
                ifs[0] : ifs[0] + npc[0] + x_p,
                ifs[1] : ifs[1] + npc[1] + y_p,
                ifs[2] : ifs[2] + npc[2] + 1,
            ] = mesh
    return meshout


def _try_text(file: Path, elt: Element) -> str:
    """Try getting text of element or raise a ParsingError."""
    text = elt.text
    if text is None:
        raise ParsingError(file, f"Element {elt} has no 'text'")
    return text


def _ifile_isnap(file: Path, elt: Element) -> tuple[int, int]:
    """Extract ifile and isnap from H5 file name/dataset."""
    data_text = _try_text(file, elt)
    h5file, group = data_text.strip().split(":/", 1)
    isnap = int(group[-5:])
    ifile = int(h5file[-14:-9])
    return ifile, isnap


def _count_subdomains(xs: XmlStream, i0_yin: int) -> tuple[range, range]:
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


@dataclass(frozen=True)
class FieldSub:
    file: Path
    dataset: str
    icore: int
    iblock: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class XmfEntry:
    isnap: int
    time: Optional[float]
    mo_lambda: Optional[float]
    mo_thick_sol: Optional[float]
    yin_yang: bool
    twod: Optional[str]
    coord_filepattern: str
    coord_shape: tuple[int, ...]
    range_yin: range
    range_yang: range
    fields: Mapping[str, tuple[int, tuple[int, ...]]]

    def coord_files_yin(self, path_root: Path) -> Iterator[Path]:
        for icore in self.range_yin:
            yield path_root / self.coord_filepattern.format(icore=icore + 1)

    def _fsub(self, path_root: Path, name: str, icore: int, yang: bool) -> FieldSub:
        ifile, shape = self.fields[name]
        yy_tag = "2" if yang else "_"
        ic = icore + 1
        return FieldSub(
            file=path_root / f"{name}_{ifile:05d}_{ic:05d}.h5",
            dataset=f"{name}{yy_tag}{ic:05d}_{self.isnap:05d}",
            icore=icore,
            iblock=int(yang),
            shape=shape,
        )

    def field_subdomains(self, path_root: Path, name: str) -> Iterator[FieldSub]:
        if name not in self.fields:
            return
        for icore in self.range_yin:
            yield self._fsub(path_root, name, icore, False)
        for icore in self.range_yang:
            yield self._fsub(path_root, name, icore, True)


@dataclass(frozen=True)
class FieldXmf:
    path: Path

    def _get_dims(self, elt: Element) -> tuple[int, ...]:
        dims = elt.attrib["Dimensions"].split()
        return tuple(map(int, dims))

    @cached_property
    def _data(self) -> Mapping[int, XmfEntry]:
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
            twod = None

            xs.skip_to_tag("Geometry")
            with xs.load() as elt_geom:
                if elt_geom.get("Type") == "X_Y":
                    twod = ""
                    for data_item in elt_geom:
                        coord = _try_text(xs.filepath, data_item).strip()[-1]
                        if coord in "XYZ":
                            twod += coord
                data_item = elt_geom[0]
                data_text = _try_text(xs.filepath, data_item)
                coord_shape = self._get_dims(data_item)
                coord_filepattern = data_text.strip().split(":/", 1)[0]
                coord_file_chunks = coord_filepattern.split("_")
                coord_file_chunks[-2] = "{icore:05d}"
                coord_filepattern = "_".join(coord_file_chunks)

            fields_info = {}
            while xs.current.tag == "Attribute":
                with xs.load() as elt_fvar:
                    name = elt_fvar.attrib["Name"]
                    elt_data = elt_fvar[0]
                    shape = self._get_dims(elt_data)
                    ifile, isnap = _ifile_isnap(xs.filepath, elt_data)
                    fields_info[name] = (ifile, shape)

            r_yin, r_yang = _count_subdomains(xs, i0_yin)

            data[isnap] = XmfEntry(
                isnap=isnap,
                time=time,
                mo_lambda=extra.get("mo_lambda"),
                mo_thick_sol=extra.get("mo_thick_sol"),
                yin_yang=yin_yang,
                twod=twod,
                coord_filepattern=coord_filepattern,
                coord_shape=coord_shape,
                range_yin=r_yin,
                range_yang=r_yang,
                fields=fields_info,
            )
        return data

    def __getitem__(self, isnap: int) -> XmfEntry:
        try:
            return self._data[isnap]
        except KeyError:
            raise ParsingError(self.path, f"no data for snapshot {isnap}")


def read_geom_h5(xdmf: FieldXmf, snapshot: int) -> dict[str, Any]:
    """Extract geometry information from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        snapshot: snapshot number.

    Returns:
        geometry information.
    """
    header: dict[str, Any] = {}

    entry = xdmf[snapshot]
    header["ti_ad"] = entry.time
    header["mo_lambda"] = entry.mo_lambda
    header["mo_thick_sol"] = entry.mo_thick_sol
    header["ntb"] = 2 if entry.yin_yang else 1

    all_meshes: list[dict[str, NDArray]] = []
    for h5file in entry.coord_files_yin(xdmf.path.parent):
        all_meshes.append({})
        with h5py.File(h5file, "r") as h5f:
            for coord, mesh in h5f.items():
                # for some reason, the array is transposed!
                all_meshes[-1][coord] = mesh[()].reshape(entry.coord_shape).T
                all_meshes[-1][coord] = _make_3d(all_meshes[-1][coord], entry.twod)

    header["ncs"] = _ncores(all_meshes, entry.twod)
    header["nts"] = list(
        (all_meshes[0]["X"].shape[i] - 1) * header["ncs"][i] for i in range(3)
    )
    header["nts"] = np.array([max(1, val) for val in header["nts"]])
    meshes = _conglomerate_meshes(all_meshes, header)
    if np.any(meshes["Z"][:, :, 0] != 0):
        # spherical
        if entry.twod is not None:  # annulus geometry...
            header["x_mesh"] = np.copy(meshes["Y"])
            header["y_mesh"] = np.copy(meshes["Z"])
            header["z_mesh"] = np.copy(meshes["X"])
        else:  # YinYang, here only yin
            header["x_mesh"] = np.copy(meshes["X"])
            header["y_mesh"] = np.copy(meshes["Y"])
            header["z_mesh"] = np.copy(meshes["Z"])
        header["r_mesh"] = np.sqrt(
            header["x_mesh"] ** 2 + header["y_mesh"] ** 2 + header["z_mesh"] ** 2
        )
        header["t_mesh"] = np.arccos(header["z_mesh"] / header["r_mesh"])
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], -header["x_mesh"]) + np.pi, -1, 1
        )
        header["e1_coord"] = header["t_mesh"][:, 0, 0]
        header["e2_coord"] = header["p_mesh"][0, :, 0]
        header["e3_coord"] = header["r_mesh"][0, 0, :]
    else:
        header["e1_coord"] = meshes["X"][:, 0, 0]
        header["e2_coord"] = meshes["Y"][0, :, 0]
        header["e3_coord"] = meshes["Z"][0, 0, :]
    header["aspect"] = (
        header["e1_coord"][-1] - header["e2_coord"][0],
        header["e1_coord"][-1] - header["e2_coord"][0],
    )
    header["rcmb"] = header["e3_coord"][0]
    if header["rcmb"] == 0:
        header["rcmb"] = -1
    else:
        header["e3_coord"] = header["e3_coord"] - header["rcmb"]
    if entry.twod is None or "X" in entry.twod:
        header["e1_coord"] = header["e1_coord"][:-1]
    if entry.twod is None or "Y" in entry.twod:
        header["e2_coord"] = header["e2_coord"][:-1]
    header["e3_coord"] = header["e3_coord"][:-1]

    return header


def _to_spherical(flds: NDArray, header: dict[str, Any]) -> NDArray:
    """Convert vector field to spherical."""
    cth = np.cos(header["t_mesh"][:, :, :-1])
    sth = np.sin(header["t_mesh"][:, :, :-1])
    cph = np.cos(header["p_mesh"][:, :, :-1])
    sph = np.sin(header["p_mesh"][:, :, :-1])
    fout = np.copy(flds)
    fout[0] = cth * cph * flds[0] + cth * sph * flds[1] - sth * flds[2]
    fout[1] = sph * flds[0] - cph * flds[1]  # need to take the opposite here
    fout[2] = sth * cph * flds[0] + sth * sph * flds[1] + cth * flds[2]
    return fout


def _flds_shape(fieldname: str, header: dict[str, Any]) -> list[int]:
    """Compute shape of flds variable."""
    shp = list(header["nts"])
    shp.append(header["ntb"])
    if len(FIELD_FILES_H5.get(fieldname, [])) == 3:
        shp.insert(0, 3)
        # extra points
        header["xp"] = int(header["nts"][0] != 1)
        shp[1] += header["xp"]
        header["yp"] = int(header["nts"][1] != 1)
        shp[2] += header["yp"]
        header["zp"] = 1
        header["xyp"] = 1
    else:
        shp.insert(0, 1)
        header["xp"] = 0
        header["yp"] = 0
        header["zp"] = 0
        header["xyp"] = 0
    return shp


def _post_read_flds(flds: NDArray, header: dict[str, Any]) -> NDArray:
    """Process flds to handle sphericity."""
    if flds.shape[0] >= 3 and header["rcmb"] > 0:
        # spherical vector
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], header["x_mesh"]), -1, 1
        )
        for ibk in range(header["ntb"]):
            flds[..., ibk] = _to_spherical(flds[..., ibk], header)
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], -header["x_mesh"]) + np.pi, -1, 1
        )
    return flds


def read_field_h5(
    xdmf: FieldXmf,
    fieldname: str,
    snapshot: int,
    header: Optional[dict[str, Any]] = None,
) -> Optional[tuple[dict[str, Any], NDArray]]:
    """Extract field data from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        fieldname: name of field to extract.
        snapshot: snapshot number.
        header: geometry information.

    Returns:
        geometry information and field data. None is returned if data is
            unavailable.
    """
    if header is None:
        header = read_geom_h5(xdmf, snapshot)

    npc = header["nts"] // header["ncs"]  # number of grid point per node
    flds = np.zeros(_flds_shape(fieldname, header))
    data_found = False

    for fsub in xdmf[snapshot].field_subdomains(xdmf.path.parent, fieldname):
        fld = _read_group_h5(fsub.file, fsub.dataset).reshape(fsub.shape)
        # for some reason, the field is transposed
        fld = fld.T
        shp = fld.shape

        if shp[-1] == 1 and header["nts"][0] == 1:  # YZ
            fld = fld.reshape((shp[0], 1, shp[1], shp[2]))
            if header["rcmb"] < 0:
                fld = fld[(2, 0, 1), ...]
        elif shp[-1] == 1:  # XZ
            fld = fld.reshape((shp[0], shp[1], 1, shp[2]))
            if header["rcmb"] < 0:
                fld = fld[(1, 2, 0), ...]
        elif fieldname in SFIELD_FILES_H5:
            fld = fld.reshape((1, npc[0], npc[1], 1))
        elif header["nts"][1] == 1:  # cart XZ
            fld = fld.reshape((1, shp[0], 1, shp[1]))
        ifs = [
            fsub.icore // np.prod(header["ncs"][:i]) % header["ncs"][i] * npc[i]
            for i in range(3)
        ]
        if fieldname in SFIELD_FILES_H5:
            ifs[2] = 0
            npc[2] = 1
        if header["zp"]:  # remove top row
            fld = fld[:, :, :, :-1]
        flds[
            :,
            ifs[0] : ifs[0] + npc[0] + header["xp"],
            ifs[1] : ifs[1] + npc[1] + header["yp"],
            ifs[2] : ifs[2] + npc[2],
            fsub.iblock,
        ] = fld
        data_found = True

    if flds.shape[0] == 3 and flds.shape[-1] == 2:  # YinYang vector
        # Yang grid is rotated compared to Yin grid
        flds[0, ..., 1] = -flds[0, ..., 1]
        vt = flds[1, ..., 1].copy()
        flds[1, ..., 1] = flds[2, ..., 1]
        flds[2, ..., 1] = vt
    flds = _post_read_flds(flds, header)

    if fieldname in SFIELD_FILES_H5:
        # remove z component
        flds = flds[..., 0, :]
    return (header, flds) if data_found else None


@dataclass(frozen=True)
class TracerSub:
    file: Path
    dataset: str
    icore: int
    iblock: int


@dataclass(frozen=True)
class XmfTracersEntry:
    isnap: int
    time: Optional[float]
    mo_lambda: Optional[float]
    mo_thick_sol: Optional[float]
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
                    ifile, isnap = _ifile_isnap(xs.filepath, data_item)
                    fields_info[name] = ifile

            while xs.current.tag == "Attribute":
                with xs.load() as elt_fvar:
                    name = elt_fvar.attrib["Name"]
                    ifile, _ = _ifile_isnap(xs.filepath, elt_fvar[0])
                    fields_info[name] = ifile

            r_yin, r_yang = _count_subdomains(xs, i0_yin)

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


def read_tracers_h5(xdmf: TracersXmf, infoname: str, snapshot: int) -> list[NDArray]:
    """Extract tracers data from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        infoname: name of information to extract.
        snapshot: snapshot number.

    Returns:
        Tracers data organized by attribute and block.
    """
    tra: list[list[NDArray]] = [[], []]  # [block][core]
    for tsub in xdmf[snapshot].tra_subdomains(xdmf.path.parent, infoname):
        tra[tsub.iblock].append(_read_group_h5(tsub.file, tsub.dataset))

    tra_concat: list[NDArray] = []
    for trab in tra:
        if trab:
            tra_concat.append(np.concatenate(trab))
    return tra_concat


def read_time_h5(h5folder: Path) -> Iterator[tuple[int, int]]:
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
