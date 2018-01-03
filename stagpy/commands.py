"""Definition of non-processing subcommands."""

from itertools import zip_longest
from math import ceil
from shutil import get_terminal_size
from subprocess import call
from textwrap import TextWrapper
import shlex
from . import conf, config, phyvars, __version__
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
        for opt, meta in conf[sub].defaults():
            if meta.cmd_arg ^ meta.conf_arg:
                opt += ' (c)' if meta.cmd_arg else ' (f)'
            hlp_lst.append((opt, meta.help_string))
        if hlp_lst:
            print('{}:'.format(sub))
            _pretty_print(hlp_lst, sep=' -- ',
                          text_width=min(get_terminal_size().columns, 100))
            print()


def config_cmd():
    """Configuration handling.

    Other Parameters:
        conf.config_file (:class:`pathlib.Path`): path of the config file.
        conf.config.create (bool): whether to create conf.config file.
        conf.config.update (bool): create conf.config_file. If it already
            exists, its content is read and only the missing parameters are set
            to their default value.
        conf.config.edit (bool): update conf.config_file and open it in
            conf.editor.
        conf.config.editor (str): the editor used by conf.config.edit to open
            the config file.
    """
    if not (conf.common.config or conf.config.create or conf.config.update or
            conf.config.edit):
        config_pp(conf.subs())
    if conf.config.create or conf.config.update:
        conf.create_config()
    if conf.config.edit:
        call(shlex.split('{} {}'.format(conf.config.editor,
                                        config.CONFIG_FILE)))
