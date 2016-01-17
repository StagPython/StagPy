"""definition of each subcommands"""

from __future__ import print_function

import constants
import misc
import rprof
from stag import StagyyData
import time_series

def field_cmd(args):
    """plot snapshots of scalar fields"""
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
    args.matplotback = None
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
