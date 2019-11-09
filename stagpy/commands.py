"""Definition of non-processing subcommands."""

from itertools import zip_longest
from math import ceil
from shutil import get_terminal_size
from textwrap import indent, TextWrapper
import sys

import loam.tools
import pandas

from . import conf, phyvars, __version__
from . import stagyydata
from .config import CONFIG_FILE, CONFIG_LOCAL
from .misc import baredoc


def info_cmd():
    """Print basic information about StagYY run.

    Other Parameters:
        conf.info
    """
    varlist = [var for var in conf.info.output.replace(',', ' ').split()]
    sdat = stagyydata.StagyyData()
    lsnap = sdat.snaps[-1]
    lstep = sdat.steps[-1]
    print('StagYY run in {}'.format(sdat.path))
    if lsnap.geom.threed:
        dimension = '{} x {} x {}'.format(lsnap.geom.nxtot,
                                          lsnap.geom.nytot,
                                          lsnap.geom.nztot)
    elif lsnap.geom.twod_xz:
        dimension = '{} x {}'.format(lsnap.geom.nxtot,
                                     lsnap.geom.nztot)
    else:
        dimension = '{} x {}'.format(lsnap.geom.nytot,
                                     lsnap.geom.nztot)
    if lsnap.geom.cartesian:
        print('Cartesian', dimension)
    elif lsnap.geom.cylindrical:
        print('Cylindrical', dimension)
    else:
        print('Spherical', dimension)
    print()
    for step in sdat.walk:
        is_snap = step.isnap is not None
        print('Step {}/{}'.format(step.istep, lstep.istep), end='')
        if is_snap:
            print(', snapshot {}/{}'.format(step.isnap, lsnap.isnap))
        else:
            print()
        series = step.timeinfo.loc[varlist]
        if conf.scaling.dimensional:
            series = series.copy()
            dimensions = []
            for var, val in series.iteritems():
                dim = phyvars.TIME.get(var) or phyvars.TIME_EXTRA.get(var)
                dim = dim.dim if dim is not None else '1'
                if dim == '1':
                    dimensions.append('')
                else:
                    series[var], dim = sdat.scale(val, dim)
                    dimensions.append(dim)
            series = pandas.concat(
                [series, pandas.Series(data=dimensions, index=series.index,
                                       name='dim')],
                axis=1)
        print(indent(series.to_string(header=False), '  '))
        print()


def _pretty_print(key_val, sep=': ', min_col_width=39, text_width=None):
    """Print a iterable of key/values.

    Args:
        key_val (list of (str, str)): the pairs of section names and text.
        sep (str): separator between section names and text.
        min_col_width (int): minimal acceptable column width
        text_width (int): text width to use. If set to None, will try to infer
            the size of the terminal.
    """
    if text_width is None:
        text_width = get_terminal_size().columns
    if text_width < min_col_width:
        min_col_width = text_width
    ncols = (text_width + 1) // (min_col_width + 1)
    colw = (text_width + 1) // ncols - 1
    ncols = min(ncols, len(key_val))

    wrapper = TextWrapper(width=colw)
    lines = []
    for key, val in key_val:
        if len(key) + len(sep) >= colw // 2:
            wrapper.subsequent_indent = ' '
        else:
            wrapper.subsequent_indent = ' ' * (len(key) + len(sep))
        lines.extend(wrapper.wrap('{}{}{}'.format(key, sep, val)))

    chunks = []
    for rem_col in range(ncols, 1, -1):
        isep = ceil(len(lines) / rem_col)
        while isep < len(lines) and lines[isep][0] == ' ':
            isep += 1
        chunks.append(lines[:isep])
        lines = lines[isep:]
    chunks.append(lines)
    lines = zip_longest(*chunks, fillvalue='')

    fmt = '|'.join(['{{:{}}}'.format(colw)] * (ncols - 1))
    fmt += '|{}' if ncols > 1 else '{}'
    print(*(fmt.format(*line) for line in lines), sep='\n')


def _layout(dict_vars, dict_vars_extra):
    """Print nicely [(var, description)] from phyvars."""
    desc = [(v, m.description) for v, m in dict_vars.items()]
    desc.extend((v, baredoc(m.description))
                for v, m in dict_vars_extra.items())
    _pretty_print(desc, min_col_width=26)


def var_cmd():
    """Print a list of available variables.

    See :mod:`stagpy.phyvars` where the lists of variables organized by command
    are defined.
    """
    print_all = not any(val for _, val in conf.var.opt_vals_())
    if print_all or conf.var.field:
        print('field:')
        _layout(phyvars.FIELD, phyvars.FIELD_EXTRA)
        print()
    if print_all or conf.var.sfield:
        print('surface field:')
        _layout(phyvars.SFIELD, {})
        print()
    if print_all or conf.var.rprof:
        print('rprof:')
        _layout(phyvars.RPROF, phyvars.RPROF_EXTRA)
        print()
    if print_all or conf.var.time:
        print('time:')
        _layout(phyvars.TIME, phyvars.TIME_EXTRA)
        print()
    if print_all or conf.var.refstate:
        print('refstate:')
        _layout(phyvars.REFSTATE, {})
        print()
    if print_all or conf.var.plates:
        print('plates:')
        _layout(phyvars.PLATES, {})


def version_cmd():
    """Print StagPy version.

    Use :data:`stagpy.__version__` to obtain the version in a script.
    """
    print('stagpy version: {}'.format(__version__))


def report_parsing_problems(parsing_out):
    """Output message about potential parsing problems."""
    _, empty, faulty = parsing_out
    if CONFIG_FILE in empty or CONFIG_FILE in faulty:
        print('Unable to read global config file', CONFIG_FILE,
              file=sys.stderr)
        print('Please run stagpy config --create',
              sep='\n', end='\n\n', file=sys.stderr)
    if CONFIG_LOCAL in faulty:
        print('Unable to read local config file', CONFIG_LOCAL,
              file=sys.stderr)
        print('Please run stagpy config --create_local',
              sep='\n', end='\n\n', file=sys.stderr)


def config_pp(subs):
    """Pretty print of configuration options.

    Args:
        subs (iterable of str): iterable with the list of conf sections to
            print.
    """
    print('(c|f): available only as CLI argument/in the config file',
          end='\n\n')
    for sub in subs:
        hlp_lst = []
        for opt, meta in conf[sub].defaults_():
            if meta.cmd_arg ^ meta.conf_arg:
                opt += ' (c)' if meta.cmd_arg else ' (f)'
            hlp_lst.append((opt, meta.help))
        if hlp_lst:
            print('{}:'.format(sub))
            _pretty_print(hlp_lst, sep=' -- ',
                          text_width=min(get_terminal_size().columns, 100))
            print()


def config_cmd():
    """Configuration handling.

    Other Parameters:
        conf.config
    """
    if not (conf.common.config or conf.config.create or
            conf.config.create_local or conf.config.update or
            conf.config.edit):
        config_pp(conf.sections_())
    loam.tools.config_cmd_handler(conf)
