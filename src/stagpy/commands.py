"""Definition of non-processing subcommands."""

from __future__ import annotations

import typing
from dataclasses import fields
from textwrap import indent

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from . import __version__, phyvars
from ._helpers import baredoc, walk
from .config import CONFIG_LOCAL, Config
from .stagyydata import _sdat_from_conf

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Callable

    from loam.base import Section

    from .datatypes import Varf, Varr, Vart


def info_cmd(conf: Config) -> None:
    """Print basic information about StagYY run."""
    sdat = _sdat_from_conf(conf.core)
    lsnap = sdat.snaps[-1]
    lstep = sdat.steps[-1]
    print(f"StagYY run in {sdat.path}")
    if lsnap.geom.threed:
        dimension = "{0.nxtot} x {0.nytot} x {0.nztot}".format(lsnap.geom)
    elif lsnap.geom.twod_xz:
        dimension = "{0.nxtot} x {0.nztot}".format(lsnap.geom)
    else:
        dimension = "{0.nytot} x {0.nztot}".format(lsnap.geom)
    if lsnap.geom.cartesian:
        print("Cartesian", dimension)
    elif lsnap.geom.cylindrical:
        print("Cylindrical", dimension)
    else:
        print("Spherical", dimension)
    print()
    for step in walk(sdat, conf):
        print(f"Step {step.istep}/{lstep.istep}", end="")
        if step.isnap is not None:
            print(f", snapshot {step.isnap}/{lsnap.isnap}")
        else:
            print()
        series = step.timeinfo.loc[list(conf.info.output)]
        print(indent(series.to_string(header=False), "  "))
        print()


def _layout(
    dict_vars: Mapping[str, Varf | Varr | Vart],
    dict_vars_extra: Mapping[str, Callable],
) -> Columns:
    """Print nicely [(var, description)] from phyvars."""
    desc = [(v, m.description) for v, m in dict_vars.items()]
    desc.extend((v, baredoc(m)) for v, m in dict_vars_extra.items())
    return Columns(
        renderables=(f"{var}: [dim]{descr}[/dim]" for var, descr in desc),
        padding=(0, 2),
        column_first=True,
    )


def var_cmd(conf: Config) -> None:
    """Print a list of available variables.

    See [stagpy.phyvars][] where the lists of variables organized by command
    are defined.
    """
    console = Console()
    print_all = not any(getattr(conf.var, fld.name) for fld in fields(conf.var))
    if print_all or conf.var.field:
        console.rule("fields", style="magenta")
        console.print(_layout(phyvars.FIELD, phyvars.FIELD_EXTRA))
        console.print()
    if print_all or conf.var.sfield:
        console.rule("surface fields", style="magenta")
        console.print(_layout(phyvars.SFIELD, {}))
        console.print()
    if print_all or conf.var.rprof:
        console.rule("radial profiles", style="magenta")
        console.print(_layout(phyvars.RPROF, phyvars.RPROF_EXTRA))
        console.print()
    if print_all or conf.var.time:
        console.rule("time series", style="magenta")
        console.print(_layout(phyvars.TIME, phyvars.TIME_EXTRA))
        console.print()
    if print_all or conf.var.refstate:
        console.rule("refstate", style="magenta")
        console.print(_layout(phyvars.REFSTATE, {}))
        console.print()


def version_cmd(conf: Config) -> None:
    """Print StagPy version.

    Use `stagpy.__version__` to obtain the version in a script.
    """
    print(f"stagpy version: {__version__}")


def config_pp(subs: Iterable[str], conf: Config) -> None:
    """Pretty print of configuration options.

    Args:
        subs: conf sections to print.
        conf: configuration.
    """
    console = Console()
    for sub in subs:
        table = Table(title=sub, box=box.SIMPLE)
        table.add_column(header="option", no_wrap=True)
        table.add_column(header="doc")
        table.add_column(header="cli", no_wrap=True)
        table.add_column(header="file", no_wrap=True)

        section: Section = getattr(conf, sub)
        for fld in fields(section):
            opt = fld.name
            entry = section.meta_(opt).entry
            table.add_row(
                opt,
                entry.doc,
                "yes" if entry.in_cli else "no",
                "yes" if entry.in_file else "no",
            )

        if table.rows:
            console.print(table)
            console.print()


def config_cmd(conf: Config) -> None:
    """Configuration handling."""
    if conf.config.create:
        conf.default_().to_file_(CONFIG_LOCAL)
    else:
        config_pp((sec.name for sec in fields(conf)), conf)
