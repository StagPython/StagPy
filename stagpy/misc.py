"""miscellaneous definitions"""

import importlib
from itertools import zip_longest
from math import ceil
import os.path
import sys

INT_FMT = '{:05d}'


def stop(*msgs):
    """print error message and exit"""
    print('ERROR:', *msgs, file=sys.stderr)
    sys.exit()


def _file_name(args, fname):
    """return full name of StagYY out file"""
    return os.path.join(args.path, args.name + '_' + fname)


def stag_file(args, fname, timestep=None, suffix=''):
    """return name of StagYY out file if exists

    specify a time step if needed
    """
    if timestep is not None:
        fname = fname + INT_FMT.format(timestep)
    fname = _file_name(args, fname + suffix)
    if not os.path.isfile(fname):
        stop('requested file {} not found'.format(fname))
    return fname


def out_name(args, par_type):
    """return out file name format for any time step"""
    return args.outname + '_' + par_type + INT_FMT


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
    fmt = _file_name(args, 't' + INT_FMT)

    endstep = 100000
    while begstep + 1 < endstep:
        guess = int(ceil((endstep + begstep) / 2))
        if os.path.isfile(fmt.format(guess)):
            begstep = guess
        else:
            endstep = guess
    return begstep


def parse_line(line, convert=None):
    """convert columns of a text line

    line values have to be space separated,
    values are converted to float by default.

    convert argument is a list of functions
    used to convert the first values.
    """
    if convert is None:
        convert = []
    line = line.split()
    for val, func in zip_longest(line, convert[:len(line)], fillvalue=float):
        yield func(val)


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
