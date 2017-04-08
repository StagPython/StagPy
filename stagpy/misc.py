"""miscellaneous definitions"""

import importlib
import sys

INT_FMT = '{:05d}'


def stop(*msgs):
    """print error message and exit"""
    print('ERROR:', *msgs, file=sys.stderr)
    sys.exit()


def out_name(args, par_type):
    """return out file name format for any time step"""
    return args.outname + '_' + par_type + INT_FMT


def set_arg(args, arg, val):
    """set a cmd line with arg string name"""
    vars(args)[arg] = val


def get_arg(args, arg):
    """set a cmd line with arg string name"""
    return vars(args)[arg]


def steps_gen(sdat, args):
    """Return generator over relevant snapshots or timesteps"""
    if args.snapshots is not None:
        return sdat.snaps[args.snapshots]
    else:
        return sdat.steps[args.timesteps]


def get_rbounds(step):
    """Radii of boundaries"""
    if step.geom is not None:
        rcmb = step.geom.rcmb
    else:
        rcmb = step.sdat.par['geometry']['r_cmb']
    rcmb = max(rcmb, 0)
    return rcmb, rcmb + 1


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
