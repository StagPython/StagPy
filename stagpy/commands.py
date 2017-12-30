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


def _layout(dict_vars, dict_vars_extra):
    """Print nicely [(var, description)] from phyvars"""
    desc = [(v, m.description) for v, m in dict_vars.items()]
    desc.extend((v, baredoc(m.description))
                for v, m in dict_vars_extra.items())
    termw = get_terminal_size().columns
    ncols = (termw + 1) // 27  # min width of 26
    colw = (termw + 1) // ncols - 1
    ncols = min(ncols, len(desc))

    wrapper = TextWrapper(width=colw)
    lines = []
    for varname, description in desc:
        wrapper.subsequent_indent = ' ' * (len(varname) + 2)
        lines.extend(wrapper.wrap('{}: {}'.format(varname, description)))

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
    fmt += '|{}'
    print(*(fmt.format(*line) for line in lines), sep='\n')


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


def config_cmd():
    """Configuration handling.

    Other Parameters:
        conf.config_file (:class:`pathlib.Path`): path of the config file.
        conf.config.create (bool): whether to create conf.config file.
        conf.config.update (bool): create conf.config_file. If it already exists,
            its content is read and only the missing parameters are set to their
            default value.
        conf.config.edit (bool): update conf.config_file and open it in
            conf.editor.
        conf.editor (str): the editor used by conf.config.edit to open the
            config file.
    """
    if not (conf.config.create or conf.config.update or conf.config.edit):
        conf.config.update = True
    if conf.config.create or conf.config.update:
        conf.create_config()
    if conf.config.edit:
        call(shlex.split('{} {}'.format(conf.config.editor,
                                        config.CONFIG_FILE)))
