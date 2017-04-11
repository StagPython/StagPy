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


def fmttime(tin):
    """Time formatting for labels"""
    aaa, bbb = '{:.2e}'.format(tin).split('e')
    bbb = int(bbb)
    return r'$t={} \times 10^{{{}}}$'.format(aaa, bbb)


def set_arg(args, arg, val):
    """set a cmd line with arg string name"""
    vars(args)[arg] = val


def get_arg(args, arg):
    """set a cmd line with arg string name"""
    return vars(args)[arg]


def list_of_vars(arg_plot):
    """Compute list of variables per plot

    Three nested lists:
    - variables on the same subplots;
    - subplots on the same figure;
    - figures.
    """
    lovs = [[[var for var in svars.split(',') if var]
             for svars in pvars.split('.') if svars]
            for pvars in arg_plot.split('_') if pvars]
    return [lov for lov in lovs if lov]


def set_of_vars(lovs):
    """Build set of variables from list"""
    return set(var for pvars in lovs for svars in pvars for var in svars)


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
