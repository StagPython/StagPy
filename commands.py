"""definition of each subcommands"""

from __future__ import print_function

import constants
from stag import StagyyData

def field_cmd(args):
    """plot snapshots of scalar fields"""
    for timestep in xrange(*map(int, args.timestep)):
        print("Processing timestep", timestep)
        for var in set(args.plot).intersection(constants.VAR_LIST):
            stgdat = StagyyData(args, constants.VAR_LIST[var].par, timestep)
            stgdat.plot_scalar(var)

def rprof_cmd(args):
    """plot radial profiles"""
    import rprof

def time_cmd(args):
    """plot time series"""
    import time_series

def var_cmd(_):
    """display a list of available variables"""
    print(*('{}: {}'.format(k, v.name) for k, v in constants.VAR_LIST.items()),
          sep='\n')
