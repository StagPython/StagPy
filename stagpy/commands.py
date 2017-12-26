"""definition of each subcommands"""

from inspect import getdoc
from itertools import zip_longest
from math import ceil
from shutil import get_terminal_size
from subprocess import call
from textwrap import TextWrapper
import shlex
from . import conf, config, constants, __version__
from . import field, rprof, time_series, plates, stagyydata


def field_cmd():
    """Plot scalar and vector fields"""
    field.field_cmd()


def rprof_cmd():
    """Plot radial profiles"""
    rprof.rprof_cmd()


def time_cmd():
    """Plot time series"""
    time_series.time_cmd()


def plates_cmd():
    """Plate analysis"""
    plates.plates_cmd()


def info_cmd():
    """Print basic information about StagYY run"""
    sdat = stagyydata.StagyyData(conf.core.path)
    lsnap = sdat.snaps.last
    lstep = sdat.steps.last
    lfields = []
    for fvar in constants.FIELD_VARS:
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
    """Print nicely [(var, description)] from *_VARS and *_VARS__EXTRA"""
    desc = [(v, m.description) for v, m in dict_vars.items()]
    desc.extend((v, getdoc(m.description)) for v, m in dict_vars_extra.items())
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
    """Print a list of available variables"""
    print('field:')
    _layout(constants.FIELD_VARS, constants.FIELD_VARS_EXTRA)
    print()
    print('rprof:')
    _layout(constants.RPROF_VARS, constants.RPROF_VARS_EXTRA)
    print()
    print('time:')
    _layout(constants.TIME_VARS, constants.TIME_VARS_EXTRA)
    print()
    print('plates:')
    _layout(constants.PLATES_VAR_LIST, {})


def version_cmd():
    """Print StagPy version"""
    print('stagpy version: {}'.format(__version__))


def config_cmd():
    """Configuration handling"""
    if not (conf.config.create or conf.config.update or conf.config.edit):
        conf.config.update = True
    if conf.config.create or conf.config.update:
        config.create_config()
    if conf.config.edit:
        call(shlex.split('{} {}'.format(conf.config.editor,
                                        config.CONFIG_FILE)))
