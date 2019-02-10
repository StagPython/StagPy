"""Lists of physical variables made available by StagPy.

They are organized by kind of variables (field, profiles, and time series).
EXTRA lists group variables that are not directly output by StagYY and need to
be computed from other variables.
"""

from collections import OrderedDict, namedtuple
from operator import attrgetter

from . import processing

Varf = namedtuple('Varf', ['description', 'dim'])
FIELD = OrderedDict((
    ('T', Varf('Temperature', 'K')),
    ('v1', Varf('x Velocity', 'm/s')),
    ('v2', Varf('y Velocity', 'm/s')),
    ('v3', Varf('z Velocity', 'm/s')),
    ('p', Varf('Pressure', 'Pa')),
    ('eta', Varf('Viscosity', 'Pa.s')),
    ('rho', Varf('Density', 'kg/m3')),
    ('sII', Varf('Second invariant of stress tensor', 'Pa')),
    ('sx1', Varf('1st comp. of principal stress eigenvector', 'Pa')),
    ('sx2', Varf('2nd comp. of principal stress eigenvector', 'Pa')),
    ('sx3', Varf('3rd comp. of principal stress eigenvector', 'Pa')),
    ('s1val', Varf('Principal stress eigenvalue', 'Pa')),
    ('edot', Varf('Strain rate', '1/s')),
    ('Tcond1', Varf('x Conductivity', 'W/(m.K)')),
    ('Tcond2', Varf('y Conductivity', 'W/(m.K)')),
    ('Tcond3', Varf('z Conductivity', 'W/(m.K)')),
    ('c', Varf('Composition', '1')),
    ('cFe', Varf('FeO content', '1')),
    ('wtr', Varf('Water concentration', '1')),
    ('age', Varf('Age', 's')),
    ('contID', Varf('ID of continents', '1')),
))

FIELD_EXTRA = OrderedDict((
    ('stream', Varf(processing.stream_function, 'm2/s')),
))

FIELD_FILES = OrderedDict((
    ('t', ['T']),
    ('vp', ['v1', 'v2', 'v3', 'p']),
    ('c', ['c']),
    ('eta', ['eta']),
    ('rho', ['rho']),
    ('wtr', ['wtr']),
    ('age', ['age']),
    ('nrc', ['contID']),
    ('str', ['sII']),
    ('sx', ['sx1', 'sx2', 'sx3', 's1val']),
    ('ed', ['edot']),
    ('tcond', ['Tcond1', 'Tcond2', 'Tcond3']),
))

FIELD_FILES_H5 = OrderedDict((
    ('Temperature', ['T']),
    ('Velocity', ['v1', 'v2', 'v3']),
    ('Dynamic_Pressure', ['p']),
    ('Composition', ['c']),
    ('IronContent', ['cFe']),
    ('Viscosity', ['eta']),
    ('Density', ['rho']),
    ('water', ['wtr']),
    ('Age', ['age']),
    ('ContinentNumber', ['contID']),
    ('Stress', ['sII']),
    ('PrincipalStressAxis', ['sx1', 'sx2', 'sx3', 's1val']),
    ('StrainRate', ['edot']),
))

SFIELD = OrderedDict((
    ('geoid', Varf('Geoid', 'm')),
    ('topo', Varf('Topography', 'm')),
))

SFIELD_FILES = OrderedDict((
    ('g', ['geoid']),
    ('cs', ['topo']),
))

Varr = namedtuple('Varr', ['description', 'kind', 'dim'])
RPROF = OrderedDict((
    ('r', Varr('Radial coordinate', 'Radius', 'm')),
    ('Tmean', Varr('Temperature', 'Temperature', 'K')),
    ('Tmin', Varr('Min temperature', 'Temperature', 'K')),
    ('Tmax', Varr('Max temperature', 'Temperature', 'K')),
    ('vrms', Varr('rms velocity', 'Velocity', 'm/s')),
    ('vmin', Varr('Min velocity', 'Velocity', 'm/s')),
    ('vmax', Varr('Max velocity', 'Velocity', 'm/s')),
    ('vzabs', Varr('Radial velocity', 'Velocity', 'm/s')),
    ('vzmin', Varr('Min radial velocity', 'Velocity', 'm/s')),
    ('vzmax', Varr('Max radial velocity', 'Velocity', 'm/s')),
    ('vhrms', Varr('Horizontal velocity', 'Velocity', 'm/s')),
    ('vhmin', Varr('Min horiz velocity', 'Velocity', 'm/s')),
    ('vhmax', Varr('Max horiz velocity', 'Velocity', 'm/s')),
    ('etalog', Varr('Viscosity', 'Viscosity', 'Pa')),
    ('etamin', Varr('Min viscosity', 'Viscosity', 'Pa')),
    ('etamax', Varr('Max viscosity', 'Viscosity', 'Pa')),
    ('elog', Varr('Strain rate', 'Strain rate', '1/s')),
    ('emin', Varr('Min strain rate', 'Strain rate', '1/s')),
    ('emax', Varr('Max strain rate', 'Strain rate', '1/s')),
    ('slog', Varr('Stress', 'Stress', 'Pa')),
    ('smin', Varr('Min stress', 'Stress', 'Pa')),
    ('smax', Varr('Max stress', 'Stress', 'Pa')),
    ('whrms', Varr('Horizontal vorticity', 'Vorticity', '1/s')),
    ('whmin', Varr('Min horiz vorticity', 'Vorticity', '1/s')),
    ('whmax', Varr('Max horiz vorticity', 'Vorticity', '1/s')),
    ('wzrms', Varr('Radial vorticity', 'Vorticity', '1/s')),
    ('wzmin', Varr('Min radial vorticity', 'Vorticity', '1/s')),
    ('wzmax', Varr('Max radial vorticity', 'Vorticity', '1/s')),
    ('drms', Varr('Divergence', 'Divergence', '1/s')),
    ('dmin', Varr('Min divergence', 'Divergence', '1/s')),
    ('dmax', Varr('Max divergence', 'Divergence', '1/s')),
    ('enadv', Varr('Advection', 'Heat flux', 'W/m2')),
    ('endiff', Varr('Diffusion', 'Heat flux', 'W/m2')),
    ('enradh', Varr('Radiogenic heating', 'Heat flux', 'W/m2')),
    ('enviscdiss', Varr('Viscous dissipation', 'Heat flux', 'W/m2')),
    ('enadiabh', Varr('Adiabatic heating', 'Heat flux', 'W/m2')),
    ('cmean', Varr('Concentration', 'Concentration', '1')),
    ('cmin', Varr('Min concentration', 'Concentration', '1')),
    ('cmax', Varr('Max concentration', 'Concentration', '1')),
    ('rhomean', Varr('Density', 'Density', 'kg/m3')),
    ('rhomin', Varr('Min density', 'Density', 'kg/m3')),
    ('rhomax', Varr('Max density', 'Density', 'kg/m3')),
    ('airmean', Varr('Air', 'Air', '1')),
    ('airmin', Varr('Min air', 'Air', '1')),
    ('airmax', Varr('Max air', 'Air', '1')),
    ('primmean', Varr('Primordial', 'Concentration', '1')),
    ('primmin', Varr('Min primordial', 'Concentration', '1')),
    ('primmax', Varr('Max primordial', 'Concentration', '1')),
    ('ccmean', Varr('Continental crust', 'Crust', '1')),
    ('ccmin', Varr('Min continental crust', 'Crust', '1')),
    ('ccmax', Varr('Max continental crust', 'Crust', '1')),
    ('fmeltmean', Varr('Melt fraction', 'Melt fraction', '1')),
    ('fmeltmin', Varr('Min melt fraction', 'Melt fraction', '1')),
    ('fmeltmax', Varr('Max melt fraction', 'Melt fraction', '1')),
    ('metalmean', Varr('Metal', 'Concentration', '1')),
    ('metalmin', Varr('Min metal', 'Concentration', '1')),
    ('metalmax', Varr('Max metal', 'Concentration', '1')),
    ('gsmean', Varr('Grain size', 'Grain size', 'm')),
    ('gsmin', Varr('Min grain size', 'Grain size', 'm')),
    ('gsmax', Varr('Max grain', 'Grain size', 'm')),
    ('viscdisslog', Varr('Viscous dissipation', 'Heat flux', 'W/m2')),
    ('viscdissmin', Varr('Min visc dissipation', 'Heat flux', 'W/m2')),
    ('viscdissmax', Varr('Max visc dissipation', 'Heat flux', 'W/m2')),
    ('advtot', Varr('Advection', 'Heat flux', 'W/m2')),
    ('advdesc', Varr('Downward advection', 'Heat flux', 'W/m2')),
    ('advasc', Varr('Upward advection', 'Heat flux', 'W/m2')),
))

RPROF_EXTRA = OrderedDict((
    ('redges', Varr(processing.r_edges, 'Radius', 'm')),
    ('dr', Varr(processing.delta_r, 'dr', 'm')),
    ('diff', Varr(processing.diff_prof, 'Heat flux', 'W/m2')),
    ('diffs', Varr(processing.diffs_prof, 'Heat flux', 'W/m2')),
    ('advts', Varr(processing.advts_prof, 'Heat flux', 'W/m2')),
    ('advds', Varr(processing.advds_prof, 'Heat flux', 'W/m2')),
    ('advas', Varr(processing.advas_prof, 'Heat flux', 'W/m2')),
    ('energy', Varr(processing.energy_prof, 'Heat flux', 'W/m2')),
    ('ciover', Varr(processing.init_c_overturn, 'Concentration', '1')),
    ('cfover', Varr(processing.c_overturned, 'Concentration', '1')),
    ('advth', Varr(processing.advth, 'Heat Flux', 'W/m2')),
))

Vart = namedtuple('Vart', ['description', 'kind', 'dim'])
TIME = OrderedDict((
    ('t', Vart('Time', 'Time', 's')),
    ('ftop', Vart('Heat flux at top', 'Heat flux', 'W/m2')),
    ('fbot', Vart('Heat flux at bottom', 'Heat flux', 'W/m2')),
    ('Tmin', Vart('Min temperature', 'Temperature', 'K')),
    ('Tmean', Vart('Temperature', 'Temperature', 'K')),
    ('Tmax', Vart('Max temperature', 'Temperature', 'K')),
    ('vmin', Vart('Min velocity', 'Velocity', 'm/s')),
    ('vrms', Vart('rms velocity', 'Velocity', 'm/s')),
    ('vmax', Vart('Max velocity', 'Velocity', 'm/s')),
    ('etamin', Vart('Min viscosity', 'Viscosity', 'Pa.s')),
    ('etamean', Vart('Viscosity', 'Viscosity', 'Pa.s')),
    ('etamax', Vart('Max viscosity', 'Viscosity', 'Pa.s')),
    ('Raeff', Vart('Effective Ra', r'$\mathrm{Ra}$', '1')),
    ('Nutop', Vart('Nusselt at top', r'$\mathrm{Nu}$', '1')),
    ('Nubot', Vart('Nusselt at bot', r'$\mathrm{Nu}$', '1')),
    ('Cmin', Vart('Min concentration', 'Concentration', '1')),
    ('Cmean', Vart('Concentration', 'Concentration', '1')),
    ('Cmax', Vart('Max concentration', 'Concentration', '1')),
    ('moltenf_mean', Vart('Molten fraction', 'Fraction', '1')),
    ('moltenf_max', Vart('Max molten fraction', 'Fraction', '1')),
    ('erupt_rate', Vart('Eruption rate', 'Eruption rate', '1/s')),
    ('erupt_tot', Vart('Erupta total', 'Eruption rate', '1/s')),
    ('erupt_heat', Vart('Erupta heat', 'Eruption rate', '1/s')),
    ('entrainment', Vart('Entrainment', 'Eruption rate', '1/s')),
    ('Cmass_error', Vart('Error on Cmass', 'Error', '1')),
    ('H_int', Vart('Internal heating', 'Internal heating', 'W/m3')),
    ('r_ic', Vart('Inner core radius', 'Inner core radius', 'm')),
    ('topT_val', Vart('Temperature at top', 'Temperature', 'K')),
    ('botT_val', Vart('Temperature at bottom', 'Temperature', 'K')),
))

TIME_EXTRA = OrderedDict((
    ('dt', Vart(processing.dtime, 'dt', 's')),
    ('dTdt', Vart(processing.dt_dt, r'dT/dt', 'K/s')),
    ('ebalance', Vart(processing.ebalance, r'$\mathrm{Nu}$', '1')),
    ('mobility', Vart(processing.mobility, 'Mobility', '1')),
))

Varp = namedtuple('Varp', ['description'])
PLATES = OrderedDict((
    ('c', Varp('Composition')),
    ('eta', Varp('Viscosity')),
    ('sc', Varp('Topography')),
    ('age', Varp('Age')),
    ('str', Varp('Stress')),
    ('sx', Varp('Principal deviatoric stress')),
    ('ed', Varp('Strain rate')),
))

SCALES = {
    'm': attrgetter('length'),
    'kg/m3': attrgetter('density'),
    'K': attrgetter('temperature'),
    'W/m2': attrgetter('heat_flux'),
    'Pa': attrgetter('stress'),
    'Pa.s': attrgetter('dyn_visc'),
    's': attrgetter('time'),
    'W/(m.K)': attrgetter('th_cond'),
    'm2/s': attrgetter('th_diff'),
    'W/m3': lambda scl: scl.power / scl.length**3,
    '1/s': lambda scl: 1 / scl.time,
    'K/s': lambda scl: scl.temperature / scl.time,
    'm/s': lambda scl: scl.length / scl.time,
}

PREFIXES = ('k', 'M', 'G')
