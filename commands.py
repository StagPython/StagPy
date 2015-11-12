"""definition of each subcommands"""

from __future__ import print_function

import constants
from stag import StagyyData

def field(args):
    """plot snapshots of scalar fields"""
    for timestep in xrange(*map(int, args.timestep)):
        print("Processing timestep", timestep)
        for var in set(args.plot).intersection(constants.varlist):
            stgdat = StagyyData(args, constants.varlist[var].par, timestep)
            stgdat.plot_scalar(var)

def rprof(args):
    """plot radial profiles"""
    import rprof

def time(args):
    """plot time series"""
    import time_series

def var(_):
    """display a list of available variables"""
    print(*('{}: {}'.format(k, v.name) for k, v in constants.varlist.items()),
          sep='\n')
