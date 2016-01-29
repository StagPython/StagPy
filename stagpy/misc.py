"""miscellaneous definitions"""

import f90nml
import importlib
from math import ceil
import os.path


def file_name(args, par_type):
    """return file name format for any time step"""
    return args.name + '_' + par_type + '{:05d}'

def path_fmt(args, par_type):
    """return full path format for any time step"""
    return os.path.join(args.path, file_name(args, par_type))

def set_arg(args, arg, val):
    """set a cmd line with arg string name"""
    vars(args)[arg] = val


def get_arg(args, arg):
    """set a cmd line with arg string name"""
    return vars(args)[arg]

def lastfile(args, begstep):
    """look for the last binary file

    research based on temperature files
    """
    fmt = path_fmt(args, 't')

    endstep = 100000
    while begstep + 1 < endstep:
        guess = int(ceil((endstep + begstep) / 2))
        if os.path.isfile(fmt.format(guess)):
            begstep = guess
        else:
            endstep = guess
    return begstep

def parse_timesteps(args):
    """parse timestep argument"""
    tstp = args.timestep.split(':')
    if not tstp[0]:
        tstp[0] = '0'
    if len(tstp) == 1:
        tstp.extend(tstp)
    if not tstp[1]:
        tstp[1] = lastfile(args, int(tstp[0]))
    tstp[1] = int(tstp[1]) + 1
    if len(tstp) != 3:
        tstp = tstp[0:2] + [1]
    if not tstp[2]:
        tstp[2] = 1
    args.timestep = list(map(int, tstp))

def plot_backend(args):
    """import matplotlib and seaborn"""
    mpl = importlib.import_module('matplotlib')
    if args.matplotback:
        mpl.use(args.matplotback)
    plt = importlib.import_module('matplotlib.pyplot')
    if args.useseaborn:
        sns = importlib.import_module('seaborn')
        args.sns = sns
    if args.xkcd:
        plt.xkcd()
    args.mpl = mpl
    args.plt = plt
