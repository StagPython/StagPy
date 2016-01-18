"""definition of each subcommands"""

from __future__ import print_function

import shlex
from subprocess import call
import config
import constants
import misc
import rprof
from stag import StagyyData
import time_series

def field_cmd(args):
    """plot snapshots of scalar fields"""
    misc.parse_timesteps(args)
    misc.plot_backend(args)
    if args.plot is not None:
        for var, meta in constants.FIELD_VAR_LIST.items():
            misc.set_arg(args, meta.arg, var in args.plot)
    for timestep in xrange(*args.timestep):
        print("Processing timestep", timestep)
        for var, meta in constants.FIELD_VAR_LIST.items():
            if misc.get_arg(args, meta.arg):
                stgdat = StagyyData(args, meta.par, timestep)
                stgdat.plot_scalar(var)

def rprof_cmd(args):
    """plot radial profiles"""
    misc.parse_timesteps(args)
    misc.plot_backend(args)
    if args.plot is not None:
        for var, meta in constants.RPROF_VAR_LIST.items():
            if var in args.plot:
                misc.set_arg(args, meta.arg, True)
                misc.set_arg(args, meta.min_max, False)
            else:
                misc.set_arg(args, meta.arg, False)
            if meta.min_max and var.upper() in args.plot:
                misc.set_arg(args, meta.arg, True)
                misc.set_arg(args, meta.min_max, True)
    rprof.rprof_cmd(args)

def time_cmd(args):
    """plot time series"""
    misc.parse_timesteps(args)
    if args.compstat:
        args.matplotback = None
    misc.plot_backend(args)
    time_series.time_cmd(args)

def var_cmd(_):
    """display a list of available variables"""
    print('field:')
    print(*('{}: {}'.format(v, m.name)
        for v, m in constants.FIELD_VAR_LIST.items()), sep='\n')
    print()
    print('rprof:')
    print(*('{}: {}'.format(v, m.name)
        for v, m in constants.RPROF_VAR_LIST.items()), sep='\n')

def config_cmd(args):
    """handling of configuration"""
    if args.create or args.update:
        config.create_config()
    if args.edit:
        call(shlex.split(args.editor + ' ' + config.CONFIG_FILE))

