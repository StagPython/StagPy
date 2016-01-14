"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict, namedtuple
import argparse
import commands

Conf = namedtuple('ConfigEntry',
        ['default', 'cmd_arg', 'shortname', 'kwargs', 'help_string'])
CORE = OrderedDict((
    ('path', Conf('./', True, 'p', {},
        'StagYY output directory')),
    ('name', Conf('test', True, 'n', {},
        'StagYY generic output file name')),
    ('geometry', Conf('annulus', True, 'g', {'choices':['annulus']},
        'geometry of the domain')),
    ('timestep', Conf('100', True, 's', {},
        'timestep slice')),
    ('xkcd', Conf(False, True, None, {},
        'use the xkcd style')),
    ('pdf', Conf(False, True, None, {},
        'produce non-rasterized pdf (slow!)')),
    ('dsa', Conf(0.05, False, None, {},
        'thickness of sticky air')),
    ('fontsize', Conf(16, False, None, {},
        'font size')),
    ('matplotback', Conf('agg', False, None, {},
        'graphical backend')),
    ('useseaborn', Conf(True, False, None, {},
        'use or not seaborn')),
    ))
FIELD = OrderedDict((
    ('plot', Conf(None, True, 'o',
        {'nargs':'?', 'const':'', 'type':str},
        'specify which variable to plot')),
    ('plot_temperature', Conf(True, False, None, {},
        'temperature scalar field')),
    ('plot_pressure', Conf(True, False, None, {},
        'pressure scalar field')),
    ('plot_stream', Conf(True, False, None, {},
        'stream function scalar field')),
    ('plot_composition', Conf(False, False, None, {},
        'composition scalar field')),
    ('plot_viscosity', Conf(False, False, None, {},
        'viscosity scalar field')),
    ('plot_density', Conf(False, False, None, {},
        'density scalar field')),
    ('shrinkcb', Conf(0.5, False, None, {},
        'color bar shrink factor')),
    ))
RPROF = OrderedDict((
    ('plot', Conf(None, True, 'o',
        {'nargs':'?', 'const':'', 'type':str},
        'specify which variable to plot')),
    ('plot_grid', Conf(True, False, None, {},
        'plot grid')),
    ('plot_temperature', Conf(True, False, None, {},
        'plot temperature')),
    ('plot_minmaxtemp', Conf(False, False, None, {},
        'plot min and max temperature')),
    ('plot_velocity', Conf(True, False, None, {},
        'plot velocity')),
    ('plot_minmaxvelo', Conf(False, False, None, {},
        'plot min and max velocity')),
    ('plot_viscosity', Conf(False, False, None, {},
        'plot viscosity')),
    ('plot_minmaxvisco', Conf(False, False, None, {},
        'plot min and max viscosity')),
    ('plot_advection', Conf(True, False, None, {},
        'plot heat advction')),
    ('plot_energy', Conf(True, False, None, {},
        'plot energy')),
    ('plot_concentration', Conf(True, False, None, {},
        'plot concentration')),
    ('plot_minmaxcon', Conf(False, False, None, {},
        'plot min and max concentration')),
    ('plot_conctheo', Conf(True, False, None, {},
        'plot concentration theo')),
    ('plot_overturn_init', Conf(True, False, None, {},
        'plot overturn init')),
    ('plot_difference', Conf(True, False, None, {},
        'plot difference between T and C profs and overturned \
                version of their initial values')),
    ('linewidth', Conf(2, False, None, {},
        'line width')),
    ))
TIME = OrderedDict((
    ('compstat', Conf(True, True, None, {},
        'compute steady state statistics')),
    ))
# one could use compstat to determine whether agg can be used or not
VAR = OrderedDict((
    ))

Sub = namedtuple('Sub', ['conf_dict', 'use_core', 'func', 'help_string'])
SUB_CMDS = OrderedDict((
    ('field', Sub(FIELD, True, commands.field_cmd,
        'plot scalar fields')),
    ('rprof', Sub(RPROF, True, commands.rprof_cmd,
        'plot radial profiles')),
    ('time', Sub(TIME, True, commands.time_cmd,
        'plot temporal series')),
    ('var', Sub(VAR, False, commands.var_cmd,
        'print the list of variables')),
    ))

class Toggle(argparse.Action):

    """argparse Action to store True/False to a +/-arg"""

    def __call__(self, parser, namespace, values, option_string=None):
        """set args attribute with True/False"""
        setattr(namespace, self.dest, bool('-+'.index(option_string[0])))

def add_args(parser, conf_dict):
    """Add arguments to a parser"""
    for arg, conf in conf_dict.items():
        if not conf.cmd_arg:
            continue
        if isinstance(conf.default, bool):
            conf.kwargs.update(action=Toggle, nargs=0)
            names = ['-{}'.format(arg), '+{}'.format(arg)]
            if conf.shortname is not None:
                names.append('-{}'.format(conf.shortname))
                names.append('+{}'.format(conf.shortname))
        else:
            conf.kwargs.setdefault('type', type(conf.default))
            names = ['--{}'.format(arg)]
            if conf.shortname is not None:
                names.append('-{}'.format(conf.shortname))
        conf.kwargs.update(help=conf.help_string)
        parser.add_argument(*names, **conf.kwargs)
    parser.set_defaults(**{a: c.default for a, c in conf_dict.items()})
    return parser

def create_parser():
    """Create cmd line args parser"""
    main_parser = argparse.ArgumentParser(
        description='read and process StagYY binary data')
    subparsers = main_parser.add_subparsers()

    core_parser = argparse.ArgumentParser(add_help=False, prefix_chars='-+')
    core_parser = add_args(core_parser, CORE)
    for sub_cmd, meta in SUB_CMDS.items():
        kwargs = {'prefix_chars':'+-', 'help':meta.help_string}
        if meta.use_core:
            kwargs.update(parents=[core_parser])
        dummy_parser = subparsers.add_parser(sub_cmd, **kwargs)
        dummy_parser = add_args(dummy_parser, meta.conf_dict)
        # have to collect the right default from config file!
        # remove --plot from config at creation!
        dummy_parser.set_defaults(func=meta.func)
    return main_parser
