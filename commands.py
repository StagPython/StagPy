"""definition of each subcommands"""

from __future__ import print_function

import constants
import rprof
from stag import StagyyData
import time_series

def field_cmd(args):
    """plot snapshots of scalar fields"""
    for timestep in xrange(*args.timestep):
        print("Processing timestep", timestep)
        for var in set(args.plot).intersection(constants.VAR_LIST):
            stgdat = StagyyData(args, constants.VAR_LIST[var].par, timestep)
            stgdat.plot_scalar(var)

def rprof_cmd(args):
    """plot radial profiles"""
    rprof.rprof_cmd(args)

def time_cmd(args):
    """plot time series"""
    time_series.time_cmd(args)

def var_cmd(_):
    """display a list of available variables"""
    print(*('{}: {}'.format(k, v.name) for k, v in constants.VAR_LIST.items()),
          sep='\n')
