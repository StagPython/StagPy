"""Lists of physical variables made available by StagPy.

They are organized by kind of variables (field, profiles, and time series).
EXTRA lists group variables that are not directly output by StagYY and need to
be computed from other variables.
"""

from collections import OrderedDict, namedtuple

from . import processing

Varf = namedtuple('Varf', ['description', 'shortname', 'popts'])
FIELD = OrderedDict((
    ('T', Varf('Temperature', 'T', {})),
    ('v1', Varf('x Velocity', 'u', {})),
    ('v2', Varf('y Velocity', 'v', {})),
    ('v3', Varf('z Velocity', 'w', {})),
    ('p', Varf('Pressure', 'p', {})),
    ('eta', Varf('Viscosity', r'\eta', {'cmap': 'viridis_r'})),
    ('rho', Varf('Density', r'\rho', {'cmap': 'RdBu'})),
    ('sII', Varf('Second invariant of stress tensor', r'\sigma_{II}',
                 {'cmap': 'plasma_r'})),
    ('sx1', Varf('1st comp. of principal stress eigenvector', 'x_1', {})),
    ('sx2', Varf('2nd comp. of principal stress eigenvector', 'x_2', {})),
    ('sx3', Varf('3rd comp. of principal stress eigenvector', 'x_3', {})),
    ('s1val', Varf('Principal stress eigenvalue', r'\sigma_1', {})),
    ('edot', Varf('Strain rate', r'\dot\varepsilon', {'cmap': 'Reds'})),
    ('Tcond1', Varf('x Conductivity', 'k_x', {})),
    ('Tcond2', Varf('y Conductivity', 'k_y', {})),
    ('Tcond3', Varf('z Conductivity', 'k_z', {})),
    ('c', Varf('Composition', 'c', {})),
    ('wtr', Varf('Water concentration', r'c_{\rm H_2O}', {})),
    ('age', Varf('Age', 'a', {})),
    ('contID', Varf('ID of continents', 'id', {})),
))

FIELD_EXTRA = OrderedDict((
    ('stream', Varf(processing.stream_function, r'\psi', {})),
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
    ('Viscosity', ['eta']),
    ('Density', ['rho']),
    ('water', ['wtr']),
    ('Age', ['age']),
    ('ContinentNumber', ['contID']),
    ('Stress', ['sII']),
    ('PrincipalStressAxis', ['sx1', 'sx2', 'sx3', 's1val']),
    ('StrainRate', ['edot']),
))

Varr = namedtuple('Varr', ['description', 'shortname'])
RPROF = OrderedDict((
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

RPROF_EXTRA = OrderedDict((
    ('redges', Varr(processing.r_edges, 'r')),
    ('dr', Varr(processing.delta_r, 'dr')),
    ('diff', Varr(processing.diff_prof, 'q')),
    ('diffs', Varr(processing.diffs_prof, 'q')),
    ('advts', Varr(processing.advts_prof, 'q')),
    ('advds', Varr(processing.advds_prof, 'q')),
    ('advas', Varr(processing.advas_prof, 'q')),
    ('energy', Varr(processing.energy_prof, 'q')),
    ('ciover', Varr(processing.init_c_overturn, 'c')),
    ('cfover', Varr(processing.c_overturned, 'c')),
))

Vart = namedtuple('Vart', ['description', 'shortname'])
TIME = OrderedDict((
    ('t', Vart('Time', 't')),
    ('ftop', Vart('Heat flux at top', 'q')),
    ('fbot', Vart('Heat flux at bottom', 'q')),
    ('Tmin', Vart('Min temperature', 'T')),
    ('Tmean', Vart('Temperature', 'T')),
    ('Tmax', Vart('Max temperature', 'T')),
    ('vmin', Vart('Min velocity', 'v')),
    ('vrms', Vart('rms velocity', 'v')),
    ('vmax', Vart('Max velocity', 'v')),
    ('etamin', Vart('Min viscosity', r'\eta')),
    ('etamean', Vart('Viscosity', r'\eta')),
    ('etamax', Vart('Max viscosity', r'\eta')),
    ('Raeff', Vart('Effective Ra', r'\mathrm{Ra}')),
    ('Nutop', Vart('Nusselt at top', r'\mathrm{Nu}')),
    ('Nubot', Vart('Nusselt at bot', r'\mathrm{Nu}')),
    ('Cmin', Vart('Min concentration', 'c')),
    ('Cmean', Vart('Concentration', 'c')),
    ('Cmax', Vart('Max concentration', 'c')),
    ('moltenf_mean', Vart('Molten fraction', 'f')),
    ('moltenf_max', Vart('Max molten fraction', 'f')),
    ('erupt_rate', Vart('Eruption rate', 'e')),
    ('erupt_tot', Vart('Erupta total', 'e')),
    ('erupt_heat', Vart('Erupta heat', 'e')),
    ('entrainment', Vart('Entrainment', 'e')),
    ('Cmass_error', Vart('Error on Cmass', r'\varepsilon')),
    ('H_int', Vart('Internal heating', 'H')),
    ('r_ic', Vart('Inner core radius', 'r_{ic}')),
    ('topT_val', Vart('Temperature at top', 'T')),
    ('botT_val', Vart('Temperature at bottom', 'T')),
))

TIME_EXTRA = OrderedDict((
    ('dTdt', Vart(processing.dt_dt, r'dT/dt')),
    ('ebalance', Vart(processing.ebalance, r'\mathrm{Nu}')),
    ('mobility', Vart(processing.mobility, 'M')),
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
