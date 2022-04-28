"""Define configuration variables for StagPy.

See :mod:`stagpy.args` for additional definitions related to the command line
interface.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union, Optional, Dict

from loam.base import entry, Section, ConfigBase
from loam import tools
from loam.tools import switch_opt, command_flag, path_entry
import loam.parsers


_index_collection = loam.parsers.tuple_of(loam.parsers.slice_or_int_parser)
_float_list = loam.parsers.tuple_of(float)

HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'
CONFIG_FILE = CONFIG_DIR / 'config.toml'
CONFIG_LOCAL = Path('.stagpy.toml')


@dataclass
class Common(Section):
    """Options used by all commands."""

    config: bool = command_flag("print config options")


@dataclass
class Core(Section):
    """Options used by most commands."""

    path: Path = path_entry(path=".", cli_short='p',
                            doc="path of StagYY run directory or par file")
    outname: str = entry(val="stagpy", cli_short='n',
                         doc='output file name prefix')
    shortname: bool = switch_opt(
        False, None, "output file name is only prefix")
    timesteps: Optional[Sequence[Union[int, slice]]] = entry(
        val_factory=lambda: None, cli_short='t', in_file=False,
        doc="timesteps slice",
        cli_kwargs={'nargs': '?', 'const': ''}, from_str=_index_collection)
    snapshots: Optional[Sequence[Union[int, slice]]] = entry(
        val_factory=lambda: None, cli_short='s', in_file=False,
        doc="snapshots slice",
        cli_kwargs={'nargs': '?', 'const': ''}, from_str=_index_collection)


@dataclass
class Plot(Section):
    """Options to tweak plots."""

    ratio: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="force aspect ratio of field plot",
        cli_kwargs={'nargs': '?', 'const': 0.6})
    raster: bool = switch_opt(True, None, "rasterize field plots")
    format: str = entry(val="pdf", doc="figure format (pdf, eps, svg, png)")
    vmin: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="minimal value on plot")
    vmax: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="maximal value on plot")
    cminmax: bool = switch_opt(False, 'C', 'constant min max across plots')
    isolines: Optional[Sequence[float]] = entry(
        val_factory=lambda: None, in_file=False, from_str=_float_list,
        doc="arbitrary isoline value, comma separated")
    mplstyle: str = entry(
        val="stagpy-paper", in_file=False,
        cli_kwargs={'nargs': '?', 'const': ''},
        doc="matplotlib style")
    xkcd: bool = command_flag("use the xkcd style")


@dataclass
class Scaling(Section):
    """Options regarding dimensionalization."""

    yearins: float = entry(val=3.154e7, in_cli=False, doc='year in seconds')
    ttransit: float = entry(val=1.78e15, in_cli=False,
                            doc="transit time in My")
    dimensional: bool = switch_opt(False, None, 'use dimensional units')
    time_in_y: bool = switch_opt(True, None, 'dimensional time is in year')
    vel_in_cmpy: bool = switch_opt(True, None,
                                   "dimensional velocity is in cm/year")
    factors: Dict[str, str] = entry(
        val_factory=lambda: {'s': 'M', 'm': 'k', 'Pa': 'G'},
        in_cli=False, doc="custom factors")


@dataclass
class Field(Section):
    """Options of the field command."""

    plot: str = entry(
        val='T,stream', cli_short='o',
        cli_kwargs={'nargs': '?', 'const': '', 'type': str},
        doc="variables to plot (see stagpy var)")
    perturbation: bool = switch_opt(
        False, None, "plot departure from average profile")
    shift: Optional[int] = entry(
        val_factory=lambda: None, in_file=False, from_str=int,
        doc="shift plot horizontally")
    timelabel: bool = switch_opt(False, None, "add label with time")
    interpolate: bool = switch_opt(False, None, "apply Gouraud shading")
    colorbar: bool = switch_opt(True, None, "add color bar to plot")
    ix: Optional[int] = entry(
        val_factory=lambda: None, in_file=False, from_str=int,
        doc="x-index of slice for 3D fields")
    iy: Optional[int] = entry(
        val_factory=lambda: None, in_file=False, from_str=int,
        doc="y-index of slice for 3D fields")
    iz: Optional[int] = entry(
        val_factory=lambda: None, in_file=False, from_str=int,
        doc="z-index of slice for 3D fields")
    isocolors: str = entry(
        val="", doc="comma-separated list of colors for isolines")
    cmap: Dict[str, str] = entry(
        val_factory=lambda: {
            'T': 'RdBu_r',
            'eta': 'viridis_r',
            'rho': 'RdBu',
            'sII': 'plasma_r',
            'edot': 'Reds'},
        in_cli=False, doc="custom colormaps")


@dataclass
class Rprof(Section):
    """Options of the rprof command."""

    plot: str = entry(
        val="Tmean", cli_short='o',
        cli_kwargs={'nargs': '?', 'const': ''},
        doc="variables to plot (see stagpy var)")
    style: str = entry(val='-', doc="matplotlib line style")
    average: bool = switch_opt(False, 'a', 'plot temporal average')
    grid: bool = switch_opt(False, 'g', 'plot grid')
    depth: bool = switch_opt(False, 'd', 'depth as vertical axis')


@dataclass
class Time(Section):
    """Options of the time command."""

    plot: str = entry(
        val="Nutop,ebalance,Nubot.Tmean", cli_short='o',
        cli_kwargs={'nargs': '?', 'const': ''},
        doc="variables to plot (see stagpy var)")
    style: str = entry(val='-', doc="matplotlib line style")
    compstat: str = entry(
        val='', in_file=False,
        cli_kwargs={'nargs': '?', 'const': ''},
        doc="compute mean and rms of listed variables")
    tstart: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="beginning time")
    tend: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="end time")
    fraction: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="ending fraction of series to process")
    marktimes: Sequence[float] = entry(
        val_str="", in_file=False, cli_short='M', from_str=_float_list,
        doc="list of times where to put a mark")
    marksteps: Sequence[Union[int, slice]] = entry(
        val_str="", cli_short='T', in_file=False, from_str=_index_collection,
        doc="list of steps where to put a mark")
    marksnaps: Sequence[Union[int, slice]] = entry(
        val_str="", cli_short='S', in_file=False, from_str=_index_collection,
        doc="list of snaps where to put a mark")


@dataclass
class Refstate(Section):
    """Options of the refstate command."""

    plot: str = entry(
        val='T', cli_short='o', cli_kwargs={'nargs': '?', 'const': ''},
        doc="variables to plot (see stagpy var)")
    style: str = entry(val='-', doc="matplotlib line style")


@dataclass
class Plates(Section):
    """Options of the plates command."""

    plot: str = entry(
        val='c.T.v2-v2.dv2-v2.topo_top', cli_short='o',
        cli_kwargs={'nargs': '?', 'const': ''},
        doc="variables to plot, can be a surface field, field, or dv2")
    field: str = entry(val='eta', doc="field to plot with plates info")
    stress: bool = switch_opt(
        False, None,
        "plot deviatoric stress instead of velocity on field plots")
    continents: bool = switch_opt(True, None, "whether to shade continents")
    vzratio: float = entry(
        val=0., doc="Ratio of mean vzabs used as threshold for plates limits")
    nbplates: bool = switch_opt(
        False, None, "plot number of plates as function of time")
    distribution: bool = switch_opt(
        False, None, "plot plate size distribution")
    zoom: Optional[float] = entry(
        val_factory=lambda: None, in_file=False, from_str=float,
        doc="zoom around surface")


@dataclass
class Info(Section):
    """Options of the info command."""

    output: str = entry(val='t,Tmean,vrms,Nutop,Nubot', cli_short='o',
                        doc="time series to print")


@dataclass
class Var(Section):
    """Options of the var command."""

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
