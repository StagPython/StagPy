"""Define configuration variables for StagPy.

See :mod:`stagpy.args` for additional definitions related to the command line
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import loam.parsers as lprs
from loam import tools
from loam.base import ConfigBase, Section, entry
from loam.collections import MaybeEntry, TupleEntry
from loam.tools import command_flag, path_entry, switch_opt

_indices = TupleEntry(inner_from_toml=lprs.slice_or_int_parser)
_plots = TupleEntry.wrapping(
    TupleEntry.wrapping(TupleEntry(str), str_sep="."), str_sep="-"
)

HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / ".config" / "stagpy"
CONFIG_FILE = CONFIG_DIR / "config.toml"
CONFIG_LOCAL = Path(".stagpy.toml")


@dataclass
class Common(Section):
    """General options."""

    config: bool = command_flag("print config options")


@dataclass
class Core(Section):
    """Core control."""

    path: Path = path_entry(
        path=".", cli_short="p", doc="path of StagYY run directory or par file"
    )
    outname: str = entry(val="stagpy", cli_short="n", doc="output file name prefix")
    shortname: bool = switch_opt(False, None, "output file name is only prefix")
    timesteps: Sequence[Union[int, slice]] = _indices.entry(
        doc="timesteps slice", in_file=False, cli_short="t"
    )
    snapshots: Sequence[Union[int, slice]] = _indices.entry(
        default=[-1], doc="snapshots slice", in_file=False, cli_short="s"
    )


@dataclass
class Plot(Section):
    """Plotting."""

    ratio: Optional[float] = MaybeEntry(float).entry(
        doc="force aspect ratio of field plot", in_file=False
    )
    raster: bool = switch_opt(True, None, "rasterize field plots")
    format: str = entry(val="pdf", doc="figure format (pdf, eps, svg, png)")
    vmin: Optional[float] = MaybeEntry(float).entry(
        doc="minimal value on plot", in_file=False
    )
    vmax: Optional[float] = MaybeEntry(float).entry(
        doc="maximal value on plot", in_file=False
    )
    cminmax: bool = switch_opt(False, "C", "constant min max across plots")
    isolines: Sequence[float] = TupleEntry(float).entry(
        doc="list of isoline values", in_file=False
    )
    mplstyle: Sequence[str] = TupleEntry(str).entry(
        default="stagpy-paper", doc="list of matplotlib styles", in_file=False
    )
    xkcd: bool = command_flag("use the xkcd style")


@dataclass
class Scaling(Section):
    """Dimensionalization."""

    yearins: float = entry(val=3.154e7, in_cli=False, doc="year in seconds")
    ttransit: float = entry(val=1.78e15, in_cli=False, doc="transit time in My")
    dimensional: bool = switch_opt(False, None, "use dimensional units")
    time_in_y: bool = switch_opt(True, None, "dimensional time is in year")
    vel_in_cmpy: bool = switch_opt(True, None, "dimensional velocity is in cm/year")
    factors: Dict[str, str] = entry(
        val_factory=lambda: {"s": "M", "m": "k", "Pa": "G"},
        in_cli=False,
        doc="custom factors",
    )


@dataclass
class Field(Section):
    """Field command."""

    plot: Sequence[Sequence[Sequence[str]]] = _plots.entry(
        default="T,stream", cli_short="o", doc="variables to plot (see stagpy var)"
    )
    perturbation: bool = switch_opt(False, None, "plot departure from average profile")
    shift: Optional[int] = MaybeEntry(int).entry(
        doc="shift plot horizontally", in_file=False
    )
    timelabel: bool = switch_opt(False, None, "add label with time")
    colorbar: bool = switch_opt(True, None, "add color bar to plot")
    ix: Optional[int] = MaybeEntry(int).entry(
        doc="x-index of slice for 3D fields", in_file=False
    )
    iy: Optional[int] = MaybeEntry(int).entry(
        doc="y-index of slice for 3D fields", in_file=False
    )
    iz: Optional[int] = MaybeEntry(int).entry(
        doc="z-index of slice for 3D fields", in_file=False
    )
    isocolors: Sequence[str] = TupleEntry(str).entry(doc="list of colors for isolines")
    cmap: Dict[str, str] = entry(
        val_factory=lambda: {
            "T": "RdBu_r",
            "eta": "viridis_r",
            "rho": "RdBu",
            "sII": "plasma_r",
            "edot": "Reds",
        },
        in_cli=False,
        doc="custom colormaps",
    )


@dataclass
class Rprof(Section):
    """Rprof command."""

    plot: Sequence[Sequence[Sequence[str]]] = _plots.entry(
        default="Tmean", cli_short="o", doc="variables to plot (see stagpy var)"
    )
    style: str = entry(val="-", doc="matplotlib line style")
    average: bool = switch_opt(False, "a", "plot temporal average")
    grid: bool = switch_opt(False, "g", "plot grid")
    depth: bool = switch_opt(False, "d", "depth as vertical axis")


@dataclass
class Time(Section):
    """Time command."""

    plot: Sequence[Sequence[Sequence[str]]] = _plots.entry(
        default="Nutop,ebalance,Nubot.Tmean",
        cli_short="o",
        doc="variables to plot (see stagpy var)",
    )
    style: str = entry(val="-", doc="matplotlib line style")
    compstat: Sequence[str] = TupleEntry(str).entry(
        doc="compute mean and rms of listed variables", in_file=False
    )
    tstart: Optional[float] = MaybeEntry(float).entry(
        doc="beginning time", in_file=False
    )
    tend: Optional[float] = MaybeEntry(float).entry(doc="end time", in_file=False)
    fraction: Optional[float] = MaybeEntry(float).entry(
        doc="ending fraction of series to process", in_file=False
    )
    marktimes: Sequence[float] = TupleEntry(float).entry(
        doc="list of times where to put a mark", in_file=False, cli_short="M"
    )
    marksteps: Sequence[Union[int, slice]] = _indices.entry(
        doc="list of steps where to put a mark", in_file=False, cli_short="T"
    )
    marksnaps: Sequence[Union[int, slice]] = _indices.entry(
        doc="list of snaps where to put a mark", in_file=False, cli_short="S"
    )


@dataclass
class Refstate(Section):
    """Refstate command."""

    plot: Sequence[str] = TupleEntry(str).entry(
        default="T", cli_short="o", doc="variables to plot (see stagpy var)"
    )
    style: str = entry(val="-", doc="matplotlib line style")


@dataclass
class Plates(Section):
    """Plates command."""

    plot: Sequence[Sequence[Sequence[str]]] = _plots.entry(
        default="c.T.v2-v2.dv2-v2.topo_top",
        cli_short="o",
        doc="variables to plot, can be a surface field, field, or dv2",
    )
    field: str = entry(val="eta", doc="field to plot with plates info")
    stress: bool = switch_opt(
        False, None, "plot deviatoric stress instead of velocity on field plots"
    )
    continents: bool = switch_opt(True, None, "whether to shade continents")
    vzratio: float = entry(
        val=0.0, doc="Ratio of mean vzabs used as threshold for plates limits"
    )
    nbplates: bool = switch_opt(
        False, None, "plot number of plates as function of time"
    )
    distribution: bool = switch_opt(False, None, "plot plate size distribution")
    zoom: Optional[float] = MaybeEntry(float).entry(
        doc="zoom around surface", in_file=False
    )


@dataclass
class Info(Section):
    """Info command."""

    output: Sequence[str] = TupleEntry(str).entry(
        default="t,Tmean,vrms,Nutop,Nubot", cli_short="o", doc="time series to print"
    )


@dataclass
class Var(Section):
    """Var command."""

    field: bool = command_flag("print field variables")
    sfield: bool = command_flag("print surface field variables")
    rprof: bool = command_flag("print rprof variables")
    time: bool = command_flag("print time variables")
    refstate: bool = command_flag("print refstate variables")


@dataclass
class Config(ConfigBase):
    """StagPy configuration."""

    common: Common
    core: Core
    plot: Plot
    scaling: Scaling
    field: Field
    rprof: Rprof
    time: Time
    refstate: Refstate
    plates: Plates
    info: Info
    var: Var
    config: tools.ConfigSection
