"""Lists of physical variables made available by StagPy.

They are organized by kind of variables (field, profiles, and time series).
EXTRA lists group variables that are not directly output by StagYY and need to
be computed from other variables.
"""

from collections import OrderedDict, namedtuple
from operator import attrgetter

from . import processing

Varf = namedtuple('Varf', ['description', 'kind'])
FIELD = OrderedDict((
    ('T', Varf('Temperature', 'Temperature')),
    ('v1', Varf('x Velocity', 'Velocity')),
    ('v2', Varf('y Velocity', 'Velocity')),
    ('v3', Varf('z Velocity', 'Velocity')),
    ('p', Varf('Pressure', 'Stress')),
    ('eta', Varf('Viscosity', 'Viscosity')),
    ('rho', Varf('Density', 'Density')),
    ('sII', Varf('Second invariant of stress tensor', 'Stress')),
    ('sx1', Varf('1st comp. of principal stress eigenvector', 'Stress')),
    ('sx2', Varf('2nd comp. of principal stress eigenvector', 'Stress')),
    ('sx3', Varf('3rd comp. of principal stress eigenvector', 'Stress')),
    ('s1val', Varf('Principal stress eigenvalue', 'Stress')),
    ('edot', Varf('Strain rate', 'Rate')),
    ('Tcond1', Varf('x Conductivity', 'Conductivity')),
    ('Tcond2', Varf('y Conductivity', 'Conductivity')),
    ('Tcond3', Varf('z Conductivity', 'Conductivity')),
    ('c', Varf('Composition', 'Composition')),
    ('cFe', Varf('FeO content', 'Composition')),
    ('wtr', Varf('Water concentration', 'Composition')),
    ('age', Varf('Age', 'Time')),
    ('contID', Varf('ID of continents', 'ID')),
))

FIELD_EXTRA = OrderedDict((
    ('stream', Varf(processing.stream_function, 'Stream')),
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

Varr = namedtuple('Varr', ['description', 'kind'])
RPROF = OrderedDict((
    ('r', Varr('Radial coordinate', 'Radius')),
    ('Tmean', Varr('Temperature', 'Temperature')),
    ('Tmin', Varr('Min temperature', 'Temperature')),
    ('Tmax', Varr('Max temperature', 'Temperature')),
    ('vrms', Varr('rms velocity', 'Velocity')),
    ('vmin', Varr('Min velocity', 'Velocity')),
    ('vmax', Varr('Max velocity', 'Velocity')),
    ('vzabs', Varr('Radial velocity', 'Velocity')),
    ('vzmin', Varr('Min radial velocity', 'Velocity')),
    ('vzmax', Varr('Max radial velocity', 'Velocity')),
    ('vhrms', Varr('Horizontal velocity', 'Velocity')),
    ('vhmin', Varr('Min horiz velocity', 'Velocity')),
    ('vhmax', Varr('Max horiz velocity', 'Velocity')),
    ('etalog', Varr('Viscosity', 'Viscosity')),
    ('etamin', Varr('Min viscosity', 'Viscosity')),
    ('etamax', Varr('Max viscosity', 'Viscosity')),
    ('elog', Varr('Strain rate', 'Strain rate')),
    ('emin', Varr('Min strain rate', 'Strain rate')),
    ('emax', Varr('Max strain rate', 'Strain rate')),
    ('slog', Varr('Stress', 'Stress')),
    ('smin', Varr('Min stress', 'Stress')),
    ('smax', Varr('Max stress', 'Stress')),
    ('whrms', Varr('Horizontal vorticity', 'Vorticity')),
    ('whmin', Varr('Min horiz vorticity', 'Vorticity')),
    ('whmax', Varr('Max horiz vorticity', 'Vorticity')),
    ('wzrms', Varr('Radial vorticity', 'Vorticity')),
    ('wzmin', Varr('Min radial vorticity', 'Vorticity')),
    ('wzmax', Varr('Max radial vorticity', 'Vorticity')),
    ('drms', Varr('Divergence', 'Divergence')),
    ('dmin', Varr('Min divergence', 'Divergence')),
    ('dmax', Varr('Max divergence', 'Divergence')),
    ('enadv', Varr('Advection', 'Heat flux')),
    ('endiff', Varr('Diffusion', 'Heat flux')),
    ('enradh', Varr('Radiogenic heating', 'Heat flux')),
    ('enviscdiss', Varr('Viscous dissipation', 'Heat flux')),
    ('enadiabh', Varr('Adiabatic heating', 'Heat flux')),
    ('cmean', Varr('Concentration', 'Concentration')),
    ('cmin', Varr('Min concentration', 'Concentration')),
    ('cmax', Varr('Max concentration', 'Concentration')),
    ('rhomean', Varr('Density', 'Density')),
    ('rhomin', Varr('Min density', 'Density')),
    ('rhomax', Varr('Max density', 'Density')),
    ('airmean', Varr('Air', 'Air')),
    ('airmin', Varr('Min air', 'Air')),
    ('airmax', Varr('Max air', 'Air')),
    ('primmean', Varr('Primordial', 'Concentration')),
    ('primmin', Varr('Min primordial', 'Concentration')),
    ('primmax', Varr('Max primordial', 'Concentration')),
    ('ccmean', Varr('Continental crust', 'Crust')),
    ('ccmin', Varr('Min continental crust', 'Crust')),
    ('ccmax', Varr('Max continental crust', 'Crust')),
    ('fmeltmean', Varr('Melt fraction', 'Melt fraction')),
    ('fmeltmin', Varr('Min melt fraction', 'Melt fraction')),
    ('fmeltmax', Varr('Max melt fraction', 'Melt fraction')),
    ('metalmean', Varr('Metal', 'Concentration')),
    ('metalmin', Varr('Min metal', 'Concentration')),
    ('metalmax', Varr('Max metal', 'Concentration')),
    ('gsmean', Varr('Grain size', 'Grain size')),
    ('gsmin', Varr('Min grain size', 'Grain size')),
    ('gsmax', Varr('Max grain', 'Grain size')),
    ('viscdisslog', Varr('Viscous dissipation', 'Heat flux')),
    ('viscdissmin', Varr('Min visc dissipation', 'Heat flux')),
    ('viscdissmax', Varr('Max visc dissipation', 'Heat flux')),
    ('advtot', Varr('Advection', 'Heat flux')),
    ('advdesc', Varr('Downward advection', 'Heat flux')),
    ('advasc', Varr('Upward advection', 'Heat flux')),
))

RPROF_EXTRA = OrderedDict((
    ('redges', Varr(processing.r_edges, 'Radius')),
    ('dr', Varr(processing.delta_r, 'dr')),
    ('diff', Varr(processing.diff_prof, 'Heat flux')),
    ('diffs', Varr(processing.diffs_prof, 'Heat flux')),
    ('advts', Varr(processing.advts_prof, 'Heat flux')),
    ('advds', Varr(processing.advds_prof, 'Heat flux')),
    ('advas', Varr(processing.advas_prof, 'Heat flux')),
    ('energy', Varr(processing.energy_prof, 'Heat flux')),
    ('ciover', Varr(processing.init_c_overturn, 'Concentration')),
    ('cfover', Varr(processing.c_overturned, 'Concentration')),
))

Vart = namedtuple('Vart', ['description', 'kind'])
TIME = OrderedDict((
    ('t', Vart('Time', 'Time')),
    ('ftop', Vart('Heat flux at top', 'Heat flux')),
    ('fbot', Vart('Heat flux at bottom', 'Heat flux')),
    ('Tmin', Vart('Min temperature', 'Temperature')),
    ('Tmean', Vart('Temperature', 'Temperature')),
    ('Tmax', Vart('Max temperature', 'Temperature')),
    ('vmin', Vart('Min velocity', 'Velocity')),
    ('vrms', Vart('rms velocity', 'Velocity')),
    ('vmax', Vart('Max velocity', 'Velocity')),
    ('etamin', Vart('Min viscosity', 'Viscosity')),
    ('etamean', Vart('Viscosity', 'Viscosity')),
    ('etamax', Vart('Max viscosity', 'Viscosity')),
    ('Raeff', Vart('Effective Ra', r'$\mathrm{Ra}$')),
    ('Nutop', Vart('Nusselt at top', r'$\mathrm{Nu}$')),
    ('Nubot', Vart('Nusselt at bot', r'$\mathrm{Nu}$')),
    ('Cmin', Vart('Min concentration', 'Concentration')),
    ('Cmean', Vart('Concentration', 'Concentration')),
    ('Cmax', Vart('Max concentration', 'Concentration')),
    ('moltenf_mean', Vart('Molten fraction', 'Fraction')),
    ('moltenf_max', Vart('Max molten fraction', 'Fraction')),
    ('erupt_rate', Vart('Eruption rate', 'Eruption rate')),
    ('erupt_tot', Vart('Erupta total', 'Eruption rate')),
    ('erupt_heat', Vart('Erupta heat', 'Eruption rate')),
    ('entrainment', Vart('Entrainment', 'Eruption rate')),
    ('Cmass_error', Vart('Error on Cmass', 'Error')),
    ('H_int', Vart('Internal heating', 'Internal heating')),
    ('r_ic', Vart('Inner core radius', 'Inner core radius')),
    ('topT_val', Vart('Temperature at top', 'Temperature')),
    ('botT_val', Vart('Temperature at bottom', 'Temperature')),
))

TIME_EXTRA = OrderedDict((
    ('dt', Vart(processing.dtime, 'dt')),
    ('dTdt', Vart(processing.dt_dt, r'dT/dt')),
    ('ebalance', Vart(processing.ebalance, r'$\mathrm{Nu}$')),
    ('mobility', Vart(processing.mobility, 'Mobility')),
))

Varp = namedtuple('Varp', ['description', 'kind'])
PLATES = OrderedDict((
    ('c', Varp('Composition', 'Composition')),
    ('eta', Varp('Viscosity', 'Viscosity')),
    ('sc', Varp('Topography', 'Topography')),
    ('age', Varp('Age', 'Time')),
    ('str', Varp('Stress', 'Stress')),
    ('sx', Varp('Principal deviatoric stress', 'Stress')),
    ('ed', Varp('Strain rate', 'Strain rate')),
))

SCALES = {
    'Radius': (attrgetter('length'), 'm'),
    'dr': (attrgetter('length'), 'm'),
    'Topography': (attrgetter('length'), 'm'),
    'Density': (attrgetter('density'), 'm'),
    'Temperature': (attrgetter('temperature'), 'K'),
    'Heat flux': (attrgetter('heat_flux'), 'W/m2'),
    'Stress': (attrgetter('stress'), 'Pa'),
    'Viscosity': (attrgetter('dyn_visc'), 'Pa.s'),
    'Time': (attrgetter('time'), 's'),
    'dt': (attrgetter('time'), 's'),
    'Strain rate': (lambda scl: 1 / scl.time, '1/s'),
    'dT/dt': (lambda scl: scl.temperature / scl.time, 'K/s'),
    'Velocity': (lambda scl: scl.length / scl.time, 'm/s'),
    'Divergence': (lambda scl: 1 / scl.time, '1/s'),
    'Vorticity': (lambda scl: 1 / scl.time, '1/s'),
}

PREFIXES = ('k', 'M', 'G')
