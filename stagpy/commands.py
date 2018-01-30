"""Definition of non-processing subcommands."""

from itertools import zip_longest
from math import ceil
from shutil import get_terminal_size
from textwrap import TextWrapper
import sys
import loam.tools
from . import DEBUG, conf, phyvars, __version__
from . import stagyydata
from .misc import baredoc


def info_cmd():
    """Print basic information about StagYY run."""
    sdat = stagyydata.StagyyData(conf.core.path)
    lsnap = sdat.snaps.last
    lstep = sdat.steps.last
    lfields = []
    for fvar in phyvars.FIELD:
        if lsnap.fields[fvar] is not None:
            lfields.append(fvar)
    print('StagYY run in {}'.format(sdat.path))
    print('Last timestep:',
          '  istep: {}'.format(lstep.istep),
          '  time:  {}'.format(lstep.timeinfo['t']),
          '  <T>:   {}'.format(lstep.timeinfo['Tmean']),
          sep='\n')
    print('Last snapshot (istep {}):'.format(lsnap.istep),
          '  isnap: {}'.format(lsnap.isnap),
          '  time:  {}'.format(lsnap.timeinfo['t']),
          '  output fields: {}'.format(','.join(lfields)),
          sep='\n')


def _pretty_print(key_val, sep=': ', min_col_width=39, text_width=None):
    """Print a iterable of key/values

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
    """Print nicely [(var, description)] from phyvars"""
    desc = [(v, m.description) for v, m in dict_vars.items()]
    desc.extend((v, baredoc(m.description))
                for v, m in dict_vars_extra.items())
    _pretty_print(desc, min_col_width=26)


def var_cmd():
    """Print a list of available variables.

    See :mod:`stagpy.phyvars` where the lists of variables organized by command
    are defined.
    """
    print('field:')
    _layout(phyvars.FIELD, phyvars.FIELD_EXTRA)
    print()
    print('rprof:')
    _layout(phyvars.RPROF, phyvars.RPROF_EXTRA)
    print()
    print('time:')
    _layout(phyvars.TIME, phyvars.TIME_EXTRA)
    print()
    print('plates:')
    _layout(phyvars.PLATES, {})


def version_cmd():
    """Print StagPy version.

    Use :data:`stagpy.__version__` to obtain the version in a script.
    """
    print('stagpy version: {}'.format(__version__))


def report_parsing_problems(missing_sections, missing_opts):
    """Output message about potential parsing problems."""
    need_update = False
    if missing_opts is None or missing_sections is None:
        print('Unable to read config file {}'.format(conf.config_file_),
              'Please run stagpy config --create',
              sep='\n', end='\n\n', file=sys.stderr)
        return
    for sub_cmd, missing in missing_opts.items():
        if DEBUG and missing:
            print('WARNING! Missing options in {} section of config file:'.
                  format(sub_cmd), *missing, sep='\n', end='\n\n',
                  file=sys.stderr)
        need_update |= bool(missing)
    if DEBUG and missing_sections:
        print('WARNING! Missing sections in config file:',
              *missing_sections, sep='\n', end='\n\n', file=sys.stderr)
    need_update |= bool(missing_sections)
    if need_update:
        print('Missing entries in config file!',
              'Please run stagpy config --update',
              end='\n\n', file=sys.stderr)


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
    if not (conf.common.config or conf.config.create or conf.config.update or
            conf.config.edit):
        config_pp(conf.subs_())
    loam.tools.config_cmd_handler(conf)
