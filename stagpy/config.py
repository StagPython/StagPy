"""Handle configuration of StagPy

Create the cmd line argument parser
and deal with the config file
"""

from collections import OrderedDict, namedtuple
from os.path import expanduser
import configparser
import pathlib
import sys
from .error import ConfigSectionError, ConfigOptionError

HOME_DIR = pathlib.Path(expanduser('~'))
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'
CONFIG_FILE = CONFIG_DIR / 'config'

Conf = namedtuple('ConfigEntry',
                  ['default', 'cmd_arg', 'shortname', 'kwargs',
                   'conf_arg', 'help_string'])

_CONF_DEF = OrderedDict()

_CONF_DEF['common'] = OrderedDict((
    ('config', Conf(None, True, None, {'action': 'store_true'},
                    False, 'print config options')),
    ('debug', Conf(None, True, None, {'action': 'store_true'},
                   False, 'debug mode')),
))

_CONF_DEF['core'] = OrderedDict((
    ('path', Conf('./', True, 'p', {},
                  True, 'StagYY run directory')),
    ('outname', Conf('stagpy', True, 'n', {},
                     True, 'StagPy output file name prefix')),
    ('timesteps', Conf(None, True, 't',
                       {'nargs': '?', 'const': ':', 'type': str},
                       False, 'timesteps slice')),
    ('snapshots', Conf(None, True, 's',
                       {'nargs': '?', 'const': ':', 'type': str},
                       False, 'snapshots slice')),
))

_CONF_DEF['plot'] = OrderedDict((
    ('raster', Conf(True, True, None, {},
                    True, 'rasterize field plots')),
    ('format', Conf('pdf', True, None, {},
                    True, 'figure format (pdf, eps, svg, png)')),
    ('fontsize', Conf(16, False, None, {},
                      True, 'font size')),
    ('linewidth', Conf(2, False, None, {},
                       True, 'line width')),
    ('matplotback', Conf('agg', False, None, {},
                         True, 'graphical backend')),
    ('useseaborn', Conf(False, False, None, {},
                        True, 'use or not seaborn')),
    ('xkcd', Conf(False, False, None, {},
                  True, 'use the xkcd style')),
))

_CONF_DEF['scaling'] = OrderedDict((
    ('yearins', Conf(3.154e7, False, None, {},
                     True, 'year in seconds')),
    ('ttransit', Conf(1.78e15, False, None, {},
                      True, 'transit time in My')),
    ('kappa', Conf(1.0e-6, False, None, {},
                   True, 'mantle thermal diffusivity m2/s')),
    ('length', Conf(2890.0e3, False, None, {},
                    True, 'thickness of mantle m')),
    ('viscosity', Conf(5.86e22, False, None, {},
                       True, 'reference viscosity Pa s')),
))

_CONF_DEF['field'] = OrderedDict((
    ('plot',
        Conf('T+stream', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, 'variables to plot (see stagpy var)')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
))

_CONF_DEF['rprof'] = OrderedDict((
    ('plot',
        Conf('Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'variables to plot (see stagpy var)')),
    ('average',
        Conf(False, True, 'a', {},
             True, 'plot temporal average')),
    ('grid',
        Conf(False, True, 'g', {},
             True, 'plot grid')),
))

_CONF_DEF['time'] = OrderedDict((
    ('plot',
        Conf('Nutop,ebalance,Nubot.Tmean', True, 'o',
             {'nargs': '?', 'const': ''},
             True, 'variables to plot (see stagpy var)')),
    ('compstat',
        Conf(False, True, None, {},
             True, 'compute steady state statistics')),
    ('tstart',
        Conf(None, True, None, {'type': float},
             False, 'beginning time')),
    ('tend',
        Conf(None, True, None, {'type': float},
             False, 'end time')),
))

_CONF_DEF['plates'] = OrderedDict((
    ('plot',
        Conf('c,eta,sc', True, 'o',
             {'nargs': '?', 'const': '', 'type': str},
             True, 'variables to plot (see stagpy var)')),
    ('vzcheck',
        Conf(False, True, None, {},
             True, 'activate Colin\'s version with vz checking')),
    ('timeprofile',
        Conf(False, True, None, {},
             True, 'nb of plates as function of time')),
    ('shrinkcb',
        Conf(0.5, False, None, {},
             True, 'color bar shrink factor')),
    ('zoom',
        Conf(None, True, None, {'type': float},
             False, 'zoom around surface')),
    ('topomin', Conf(-40, False, None, {},
                     True, 'min topography in plots')),
    ('topomax', Conf(100, False, None, {},
                     True, 'max topography in plots')),
    ('agemin', Conf(-50, False, None, {},
                    True, 'min age in plots')),
    ('agemax', Conf(500, False, None, {},
                    True, 'max age in plots')),
    ('vmin', Conf(-5000, False, None, {},
                  True, 'min velocity in plots')),
    ('vmax', Conf(5000, False, None, {},
                  True, 'max velocity in plots')),
    ('dvmin', Conf(-250000, False, None, {},
                   True, 'min velocity derivative in plots')),
    ('dvmax', Conf(150000, False, None, {},
                   True, 'max velocity derivative in plots')),
    ('stressmin', Conf(0, False, None, {},
                       True, 'min stress in plots')),
    ('stressmax', Conf(800, False, None, {},
                       True, 'max stress in plots')),
    ('lstressmax', Conf(50, False, None, {},
                        True, 'max lithospheric stress in plots')),
))

_CONF_DEF['info'] = OrderedDict((
))

_CONF_DEF['var'] = OrderedDict((
))

_CONF_DEF['version'] = OrderedDict((
))

_CONF_DEF['config'] = OrderedDict((
    ('create',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'create new config file')),
    ('update',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'add missing entries to config file')),
    ('edit',
        Conf(None, True, None, {'action': 'store_true'},
             False, 'open config file in a text editor')),
    ('editor',
        Conf('vim', False, None, {},
             True, 'text editor')),
))


class _SubConfig:

    """Hold options for a single subcommand"""

    def __init__(self, parent, name, defaults):
        self._parent = parent
        self._name = name
        self._def = defaults
        for opt, meta in self.defaults():
            self[opt] = meta.default

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, option):
        delattr(self, option)

    def __getattr__(self, option):
        if option in self._def:
            self[option] = self._def[option].default
        else:
            raise ConfigOptionError(option)
        return self[option]

    def __iter__(self):
        return iter(self._def.keys())

    def options(self):
        """Iterator over configuration option names.

        Yields:
            option names.
        """
        return iter(self)

    def opt_vals(self):
        """Iterator over option names and option values.

        Yields:
            tuples with option names, and option values.
        """
        for opt in self.options():
            yield opt, self[opt]

    def defaults(self):
        """Iterator over option names, and option metadata.

        Yields:
            tuples with option names, and :class:`Conf` instances holding
            option metadata.
        """
        return self._def.items()

    def read_section(self):
        """Read section of config parser

        read section corresponding to the sub command
        and set options accordingly
        """
        missing_opts = []
        config_parser = self._parent.config_parser
        for opt, meta_opt in self.defaults():
            if not meta_opt.conf_arg:
                continue
            if not config_parser.has_option(self._name, opt):
                missing_opts.append(opt)
                continue
            if isinstance(meta_opt.default, bool):
                dflt = config_parser.getboolean(self._name, opt)
            elif isinstance(meta_opt.default, float):
                dflt = config_parser.getfloat(self._name, opt)
            elif isinstance(meta_opt.default, int):
                dflt = config_parser.getint(self._name, opt)
            else:
                dflt = config_parser.get(self._name, opt)
            self[opt] = dflt
        return missing_opts


class StagpyConfiguration:

    """Hold StagPy configuration options values.

    :data:`stagpy.conf` is a global instance of this class. Instances of this
    class are set with internally defined default values and updated with the
    content of :attr:`config_file`.

    Run ``stagpy config`` to print the list of available configuration options
    with a short description. The list of configuration options is also
    available for each subcommand with the ``stagpy <cmd> --config`` flag.

    A configuration option can be accessed both with attribute and item access
    notations, these two lines access the same option value::

        stagpy.conf.core.path
        stagpy.conf['core']['path']

    To reset a configuration option (or an entire section) to its default
    value, simply delete it (with item or attribute notation)::

        del stagpy.conf['core']  # reset all core options
        del stagpy.conf.field.plot  # reset a particular option

    It will be set to its default value the next time you access it.
    """

    def __init__(self, config_file):
        """Initialization of instances:

        Args:
            config_file (pathlike): path of config file.

        Attributes:
            config_parser: :class:`configparser.ConfigParser` instance.
            config_file (pathlib.Path): path of config file.
        """
        self._def = _CONF_DEF  # defaults information
        for sub in self.subs():
            self[sub] = _SubConfig(self, sub, self._def[sub])
        self.config_parser = configparser.ConfigParser()
        if config_file is not None:
            self.config_file = pathlib.Path(config_file)
            self._missing_parsing = self.read_config()
        else:
            self.config_file = pathlib.Path('.stagpyconfig')
            self._missing_parsing = {}, []

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, sub):
        delattr(self, sub)

    def __getattr__(self, sub):
        if sub in self._def:
            self[sub] = _SubConfig(self, sub, self._def[sub])
        else:
            raise ConfigSectionError(sub)
        return self[sub]

    def __iter__(self):
        return iter(self._def.keys())

    def subs(self):
        """Iterator over configuration subsection names.

        Yields:
            subsection names.
        """
        return iter(self)

    def options(self):
        """Iterator over subsection and option names.

        This iterator is also implemented at the subsection level. The two
        loops produce the same output::

            for sub, opt in conf.options():
                print(sub, opt)

            for sub in conf.subs():
                for opt in conf[sub].options():
                    print(sub, opt)

        Yields:
            tuples with subsection and options names.
        """
        for sub in self:
            for opt in self._def[sub]:
                yield sub, opt

    def opt_vals(self):
        """Iterator over subsection, option names, and option values.

        This iterator is also implemented at the subsection level. The two
        loops produce the same output::

            for sub, opt, val in conf.opt_vals():
                print(sub, opt, val)

            for sub in conf.subs():
                for opt, val in conf[sub].opt_vals():
                    print(sub, opt, val)

        Yields:
            tuples with subsection, option names, and option values.
        """
        for sub, opt in self.options():
            yield sub, opt, self[sub][opt]

    def defaults(self):
        """Iterator over subsection, option names, and option metadata.

        This iterator is also implemented at the subsection level. The two
        loops produce the same output::

            for sub, opt, meta in conf.defaults():
                print(sub, opt, meta.default)

            for sub in conf.subs():
                for opt, meta in conf[sub].defaults():
                    print(sub, opt, meta.default)

        Yields:
            tuples with subsection, option names, and :class:`Conf`
            instances holding option metadata.
        """
        for sub, opt in self.options():
            yield sub, opt, self._def[sub][opt]

    def create_config(self):
        """Create config file.

        Create a config file at path :attr:`config_file`.

        Other Parameters:
            conf.config.update (bool): if set to True and :attr:`config_file`
                already exists, its content is read and all the options it sets
                are kept in the produced config file.
        """
        if not self.config_file.parent.exists():
            self.config_file.parent.mkdir(parents=True)
        config_parser = configparser.ConfigParser()
        for sub_cmd in self.subs():
            config_parser.add_section(sub_cmd)
            for opt, opt_meta in self[sub_cmd].defaults():
                if opt_meta.conf_arg:
                    if self.config.update:
                        val = str(self[sub_cmd][opt])
                    else:
                        val = str(opt_meta.default)
                    config_parser.set(sub_cmd, opt, val)
        with self.config_file.open('w') as out_stream:
            config_parser.write(out_stream)

    def read_config(self):
        """Read config file and set config values accordingly."""
        if not self.config_file.is_file():
            return None, None
        try:
            self.config_parser.read(str(self.config_file))
        except configparser.Error:
            return None, None
        missing_sections = []
        missing_opts = {}
        for sub in self.subs():
            if not self.config_parser.has_section(sub):
                missing_sections.append(sub)
                continue
            missing_opts[sub] = self[sub].read_section()
        return missing_opts, missing_sections

    def report_parsing_problems(self):
        """Output message about potential parsing problems.

        If there were some missing sections or options in the
        :attr:`config_file` when it was last read, this will be printed by this
        function in sys.stderr.
        """
        missing_opts, missing_sections = self._missing_parsing
        need_update = False
        if missing_opts is None or missing_sections is None:
            print('Unable to read config file {}'.format(self.config_file),
                  'Please run stagpy config --create',
                  sep='\n', end='\n\n', file=sys.stderr)
            return
        for sub_cmd, missing in missing_opts.items():
            if self.common.debug and missing:
                print('WARNING! Missing options in {} section of config file:'.
                      format(sub_cmd), *missing, sep='\n', end='\n\n',
                      file=sys.stderr)
            need_update |= bool(missing)
        if self.common.debug and missing_sections:
            print('WARNING! Missing sections in config file:',
                  *missing_sections, sep='\n', end='\n\n', file=sys.stderr)
        need_update |= bool(missing_sections)
        if need_update:
            print('Missing entries in config file!',
                  'Please run stagpy config --update',
                  end='\n\n', file=sys.stderr)
