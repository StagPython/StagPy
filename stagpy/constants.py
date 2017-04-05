"""defines some constants"""

import pathlib
from collections import OrderedDict, namedtuple
from os.path import expanduser

HOME_DIR = pathlib.Path(expanduser('~'))
CONFIG_DIR = HOME_DIR / '.config' / 'stagpy'

Varf = namedtuple('Varf', ['par', 'name', 'arg', 'pcolor_opts'])
FIELD_VAR_LIST = OrderedDict((
    ('t', Varf('t', 'Temperature', 'plot_temperature', {})),
    ('c', Varf('c', 'Composition', 'plot_composition', {})),
    ('n', Varf('eta', 'Viscosity', 'plot_viscosity',
               {'cmap': 'jet_r'})),
    ('d', Varf('rho', 'Density', 'plot_density',
               {'cmap': 'bwr_r', 'vmin': 0.96, 'vmax': 1.04})),
    ('h', Varf('wtr', 'Water', 'plot_water', {})),
    ('a', Varf('age', 'Age', 'plot_age',
               {'vmin': 0})),
    ('i', Varf('nrc', 'ID of continents', 'plot_continents', {})),
    ('s', Varf('str', 'Stress (second invariant)', 'plot_stress',
               {'cmap': 'gnuplot2_r', 'vmin': 500, 'vmax': 20000})),
    ('x', Varf('sx', 'Principal deviatoric stress', 'plot_deviatoric_stress',
               {})),
    ('e', Varf('ed', 'Strain rate', 'plot_strainrate',
               {'cmap': 'Reds', 'vmin': 500, 'vmax': 20000})),
    ('u', Varf('vp', 'x Velocity', 'plot_xvelo', {})),
    ('v', Varf('vp', 'y Velocity', 'plot_yvelo', {})),
    ('w', Varf('vp', 'z Velocity', 'plot_zvelo', {})),
    ('p', Varf('vp', 'Pressure', 'plot_pressure', {})),
    ('l', Varf('vp', 'Stream function', 'plot_stream', {})),
))

Varr = namedtuple('Varr', ['name', 'arg', 'min_max', 'prof_idx'])
RPROF_VAR_LIST = OrderedDict((
    ('t', Varr('Temperature', 'plot_temperature', 'plot_minmaxtemp', 1)),
    ('v', Varr('Vertical velocity', 'plot_velocity', 'plot_minmaxvelo', 7)),
    ('u', Varr('Horizontal velocity', 'plot_velocity', 'plot_minmaxvelo', 10)),
    ('n', Varr('Viscosity', 'plot_viscosity', 'plot_minmaxvisco', 13)),
    ('c', Varr('Concentration', 'plot_concentration', 'plot_minmaxcon', 36)),
    ('g', Varr('Grid', 'plot_grid', None, None)),
    ('z', Varr('Grid km', 'plot_grid_units', None, None)),
    ('a', Varr('Advection', 'plot_advection', None, None)),
    ('e', Varr('Energy', 'plot_energy', None, None)),
    ('h', Varr('Concentration Theo', 'plot_conctheo', None, None)),
    ('i', Varr('Init overturn', 'plot_overturn_init', None, None)),
    ('d', Varr('Difference', 'plot_difference', None, None)),
))

Varr = namedtuple('Varr', ['description', 'shortname'])
RPROF_VARS = OrderedDict((
    ('r', Varr('Radial coordinate', 'r')),
    ('Tmean', Varr('Temperature', 'T')),
    ('Tmin', Varr('Min temperature', 'T')),
    ('Tmax', Varr('Max temperature', 'T')),
    ('vrms', Varr('rms velocity', 'v')),
    ('vmin', Varr('Min velocity', 'v')),
    ('vmax', Varr('Max velocity', 'v')),
    ('vzabs', Varr('Radial velocity', 'v')),
    ('vzmin', Varr('Min radial velocity', 'v')),
    ('vzmax', Varr('Max radial velocity', 'v')),
    ('vhrms', Varr('Horizontal velocity', 'v')),
    ('vhmin', Varr('Min horiz velocity', 'v')),
    ('vhmax', Varr('Max horiz velocity', 'v')),
    ('etalog', Varr('Viscosity', r'\eta')),
    ('etamin', Varr('Min viscosity', r'\eta')),
    ('etamax', Varr('Max viscosity', r'\eta')),
    ('elog', Varr('Strain rate', r'\varepsilon')),
    ('emin', Varr('Min strain rate', r'\varepsilon')),
    ('emax', Varr('Max strain rate', r'\varepsilon')),
    ('slog', Varr('Stress', r'\sigma')),
    ('smin', Varr('Min stress', r'\sigma')),
    ('smax', Varr('Max stress', r'\sigma')),
    ('whrms', Varr('Horizontal vorticity', r'\omega')),
    ('whmin', Varr('Min horiz vorticity', r'\omega')),
    ('whmax', Varr('Max horiz vorticity', r'\omega')),
    ('wzrms', Varr('Radial vorticity', r'\omega')),
    ('wzmin', Varr('Min radial vorticity', r'\omega')),
    ('wzmax', Varr('Max radial vorticity', r'\omega')),
    ('drms', Varr('Divergence', r'\nabla\cdot u')),
    ('dmin', Varr('Min divergence', r'\nabla\cdot u')),
    ('dmax', Varr('Max divergence', r'\nabla\cdot u')),
    ('enadv', Varr('Advection', 'q')),
    ('endiff', Varr('Diffusion', 'q')),
    ('enradh', Varr('Radiogenic heating', 'q')),
    ('enviscdiss', Varr('Viscous dissipation', 'q')),
    ('enadiabh', Varr('Adiabatic heating', 'q')),
    ('cmean', Varr('Concentration', 'c')),
    ('cmin', Varr('Min concentration', 'c')),
    ('cmax', Varr('Max concentration', 'c')),
    ('rhomean', Varr('Density', r'\rho')),
    ('rhomin', Varr('Min density', r'\rho')),
    ('rhomax', Varr('Max density', r'\rho')),
    ('airmean', Varr('Air', 'c')),
    ('airmin', Varr('Min air', 'c')),
    ('airmax', Varr('Max air', 'c')),
    ('primmean', Varr('Primordial', 'c')),
    ('primmin', Varr('Min primordial', 'c')),
    ('primmax', Varr('Max primordial', 'c')),
    ('ccmean', Varr('Continental crust', 'c')),
    ('ccmin', Varr('Min continental crust', 'c')),
    ('ccmax', Varr('Max continental crust', 'c')),
    ('fmeltmean', Varr('Melt fraction', 'c')),
    ('fmeltmin', Varr('Min melt fraction', 'c')),
    ('fmeltmax', Varr('Max melt fraction', 'c')),
    ('metalmean', Varr('Metal', 'c')),
    ('metalmin', Varr('Min metal', 'c')),
    ('metalmax', Varr('Max metal', 'c')),
    ('gsmean', Varr('Grain size', 's')),
    ('gsmin', Varr('Min grain size', 's')),
    ('gsmax', Varr('Max grain', 's')),
    ('viscdisslog', Varr('Viscous dissipation', 'q')),
    ('viscdissmin', Varr('Min visc dissipation', 'q')),
    ('viscdissmax', Varr('Max visc dissipation', 'q')),
    ('advtot', Varr('Advection', 'q')),
    ('advdesc', Varr('Downward advection', 'q')),
    ('advasc', Varr('Upward advection', 'q')),
))

RPROF_VARS_EXTRA = OrderedDict((
))

TIME_VAR_LIST = [
    't', 'ftop', 'fbot', 'Tmin', 'Tmean', 'Tmax', 'vmin', 'vrms',
    'vmax', 'etamin', 'etamean', 'etamax', 'Raeff', 'Nutop', 'Nubot', 'Cmin',
    'Cmean', 'Cmax', 'moltenf_mean', 'moltenf_max', 'erupt_rate', 'erupt_tot',
    'erupt_heat', 'entrainment', 'Cmass_error', 'H_int', 'r_ic', 'topT_val',
    'botT_val']

Varp = namedtuple('Varp', ['par', 'name', 'arg'])
PLATES_VAR_LIST = OrderedDict((
    ('c', Varp('c', 'Composition', 'plot_composition')),
    ('n', Varp('eta', 'Viscosity', 'plot_viscosity')),
    ('r', Varp('cs', 'Topography', 'plot_topography')),
    ('a', Varp('age', 'Age', 'plot_age')),
    ('s', Varp('str', 'Stress', 'plot_stress')),
    ('x', Varp('sx', 'Principal deviatoric stress', 'plot_deviatoric_stress')),
    ('e', Varp('ed', 'Strain rate', 'plot_strainrate')),
))
