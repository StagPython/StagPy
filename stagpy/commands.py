"""definition of each subcommands"""

from . import constants, misc, field, rprof, time_series, plates
from . import __version__


def field_cmd(args):
    """plot snapshots of fields"""
    misc.plot_backend(args)
    if args.plot is not None:
        for var, meta in constants.FIELD_VAR_LIST.items():
            misc.set_arg(args, meta.arg, var in args.plot)
    field.field_cmd(args)


def rprof_cmd(args):
    """plot radial profiles"""
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
    if args.compstat:
        args.matplotback = None
    misc.plot_backend(args)
    time_series.time_cmd(args)


def plates_cmd(args):
    """plate analysis"""
    misc.plot_backend(args)
    if args.plot is not None:
        for var, meta in constants.PLATES_VAR_LIST.items():
            misc.set_arg(args, meta.arg, var in args.plot)
    plates.plates_cmd(args)


def var_cmd(_):
    """display a list of available variables"""
    print('field:')
    print(*('{}: {}'.format(v, m.name)
          for v, m in constants.FIELD_VAR_LIST.items()), sep='\n')
    print()
    print('rprof:')
    print(*('{}: {}'.format(v, m.name)
          for v, m in constants.RPROF_VAR_LIST.items()), sep='\n')
    print()
    print('plates:')
    print(*('{}: {}'.format(v, m.name)
          for v, m in constants.PLATES_VAR_LIST.items()), sep='\n')


def version_cmd(_):
    """print current version"""
    print('stagpy version: v{}'.format(__version__))
