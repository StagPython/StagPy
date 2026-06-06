"""Parsers of text output files."""

from __future__ import annotations

import re
import typing
from operator import itemgetter

import pandas as pd

from .._helpers import resize
from ..error import ParsingError
from ..phyvars import RPROF

if typing.TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


def tseries(timefile: Path) -> DataFrame | None:
    """Read temporal series text file.

    Args:
        timefile: path of the time.dat file.

    Returns:
        A `pandas.DataFrame` containing the time series, organized by
            variables in columns and time steps in rows.
    """
    if not timefile.is_file():
        return None

    with timefile.open() as fid:
        colnames = fid.readline().strip().split()
    # extra columns in case some were added mid-run
    resize(colnames, len(colnames) + 10)

    data = pd.read_csv(
        timefile,
        sep=r"\s+",
        dtype=str,
        header=None,
        names=colnames,
        skiprows=1,
        index_col="istep",
        engine="c",
        memory_map=True,
        on_bad_lines="skip",
    )
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")

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
    data.dropna(axis="columns", how="all", inplace=True)

    return data


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


def rprof(rproffile: Path) -> tuple[dict[int, DataFrame], DataFrame | None]:
    """Extract radial profiles data.

    Args:
        rproffile: path of the rprof.dat file.

    Returns:
        profs: a dict mapping istep to radial profiles.
        times: the time indexed by time steps.
    """
    if not rproffile.is_file():
        return {}, None

    with rproffile.open() as fid:
        colnames = fid.readline().strip().split()
    if not colnames:
        colnames = list(RPROF.keys())

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
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")

    isteps = _extract_rsnap_isteps(rproffile, data)

    all_data = {}
    for istep, _, step_df in isteps:
        step_df.index = pd.RangeIndex(step_df.shape[0])  # check whether necessary
        step_cols = list(colnames)
        resize(step_cols, step_df.shape[1])
        step_df.columns = pd.Index(step_cols)
        all_data[istep] = step_df

    df_times = pd.DataFrame(
        list(map(itemgetter(1), isteps)), index=pd.Index(map(itemgetter(0), isteps))
    )
    return all_data, df_times


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
) -> tuple[list[list[DataFrame]], list[DataFrame]] | None:
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
    data = data.apply(pd.to_numeric, raw=True, errors="coerce")
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
