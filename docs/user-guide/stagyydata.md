The StagyyData class
====================

The [`StagyyData`][stagpy.stagyydata.StagyyData] class is a generic lazy accessor to
StagYY output data (HDF5 or "legacy" format) you can use in your own Python
scripts. This section assumes the [`StagyyData`][stagpy.stagyydata.StagyyData]
instance is called `sdat`. You can create such an instance like this:

```py
from pathlib import Path

from stagpy.stagyydata import StagyyData

sdat = StagyyData(Path("path/to/run/"))
```

where `path/to/run/` is the path towards the directory containing your run
(where the `par` file is). This path can be absolute or relative to the
current working directory.

Snapshots and time steps
------------------------

A StagYY run is a succession of time steps with information such as the mean
temperature of the domain outputted at each time step. Now and then, radial
profiles and complete fields are saved, constituting a snapshot.

A [`StagyyData`][stagpy.stagyydata.StagyyData] instance has two attributes to
access time steps and snapshots in a consistent way: `StagyyData.steps` and
`StagyyData.snaps`. Accessing the `n`-th time step or the `m`-th snapshot is
done using the item access notation (square brackets):

```py
sdat.steps[n]
sdat.snaps[m]
```

These two expressions each return a [`Step`][stagpy.step.Step] instance.
Moreover, if the `m`-th snapshot was done at the `n`-th step, both
expressions return the same [`Step`][stagpy.step.Step] instance. In
other words, if for example the 100th snapshot was made at the 1000th step,
`sdat.steps[1000] is sdat.snaps[100]` is true.  The correspondence between
time steps and snapshots is deduced from available binary files.

Negative indices are allowed, `sdat.steps[-1]` being the last time step
(inferred from temporal series information) and `sdat.snaps[-1]` being the
last snapshot (inferred from available binary files).

Iterators and filters
---------------------

`StagyyData.steps` and `StagyyData.snaps` accept slices.
`sdat.steps[a:b:c]` returns a [`StepsView`][stagpy.stagyydata.StepsView]
instance. Iterating through this object is similar to iterating through the
generator `(sdat.steps[n] for n in range(a, b, c))` (negative indices are
also properly taken care of). For example, the following code process every
even snapshot:

```py
for step in sdat.snaps[::2]:
    do_something(step)
```

[`StepsView`][stagpy.stagyydata.StepsView] instances offer the possibility to
filter out the steps objects with the
[`filter`][stagpy.stagyydata.StepsView.filter] method (see its API reference
documentation for a list of available filters). For example, the following loop
process every timestep from the 512-th that has temperature and viscosity field
data available:

```py
for step in sdat.steps[512:].filter(fields=['T', 'eta']):
    do_something(step)
```

As a convenience, iterating through `sdat.steps` and `sdat.snaps` is equivalent
to iterating through `sdat.steps[:]` and `sdat.snaps[:]`. Similarly, calling
`sdat.steps.filter()` and `sdat.snaps.filter()` is a shortcut for
`sdat.steps[:].filter()` and `sdat.snaps[:].filter()`.

Parameters file
---------------

Parameters set in the `par` file are accessible through the `StagyyData.par`
attribute. For example, to access the Rayleigh number from the `refstate`
section of the par file, one can use `sdat.par.nml["refstate"]["ra0"]`.

Radial profiles
---------------

Radial profile data are accessible trough the `Step.rprofs` attribute. This
attribute implements getitem to access radial profiles.  Keys are the names of
available variables (such as e.g. `"Tmean"` and `"vzabs"`). Items are
[`Rprof`][stagpy.datatypes.Rprof] with three fields:

- `values`: the profile itself;
- `rad`: the radial position at which the profile is evaluated;
- `meta`: metadata of the profile, itself a dataclass with:

    - `description`: explanation of what the profile is;
    - `kind`: the category of profile;
    - `dim`: the dimension of the profile (if applicable) in SI units.

The list of available variables can be obtained by running

```sh title="shell"
stagpy var --rprof
```

For example, `sdat.steps[1000].rprofs["Tmean"]` is the temperature profile of
the 1000th timestep.

Time series
-----------

Temporal data are accessible through the `StagyyData.tseries` attribute. This
attribute implements getitem to access time series.  Keys are the names of
available variables (such as e.g. `"Tmean"` and `"ftop"`).  Items are
[`Tseries`][stagpy.datatypes.Tseries] with three fields:

- `values`: the series itself;
- `time`: the times at which the series is evaluated;
- `meta`: metadata of the series, itself a dataclass with:

    - `description`: explanation of what the series is;
    - `kind`: the category of series;
    - `dim`: the dimension of the series (if applicable) in SI units.

The list of available variables can be obtained by running

```sh title="shell"
stagpy var --time
```

The time series data at a given time step can be accessed from `Step.timeinfo`.
For example, `sdat.steps[1000].timeinfo` is equivalent to
`sdat.tseries.at_step(1000)`. Both are `pandas.Series` indexed by the available
variables.


Geometry
--------

Geometry information are read from binary files. `Step.geom` has various
attributes defining the geometry of the problem.

`cartesian`, `curvilinear`, `cylindrical`, `spherical` and `yinyang` booleans
define the shape of the domain (`curvilinear` being the opposite of
`cartesian`, `True` if `cylindrical` or `spherical` is `True`).

`twod_xz`, `twod_yz`, `twod` and `threed` booleans indicate the number of
spatial dimensions in the simulation. Note that fields are always four
dimensional arrays (spatial + blocks) regardless of the actual dimension of the
domain.

`nxtot`, `nytot`, `nztot`, `nbtot`, `nttot`, `nptot` and `nrtot` are the total
number of points in the various spatial directions. Note that `nttot`, `nptot`
and `nrtot` are the same as `nxtot`, `nytot` and `nztot` regardless of whether
the geometry is cartesian or curvilinear.

`x_centers`, `y_centers`, and `z_centers` as well as `t_centers`, `p_centers`,
and `r_centers` are the coordinates of cell centers in the three directions.
As for the total number of points, they are the same regardless of the actual
geometry.

Similarly to `*_centers` attributes, `x_walls`, `y_walls`, and `z_walls` as
well as `t_walls`, `p_walls`, and `r_walls` are the coordinates of cell walls
in the three directions.

Scalar and vector fields
------------------------

Vector and scalar fields are accessible through `Step.fields` using their name
as key. For example, the temperature field of the 100th snapshot is obtained
with `sdat.snaps[100].fields["T"]`.  Valid names of fields can be obtained by
running

```sh title="shell"
stagpy var --field
```

Items have two elements:

- `values`: the field itself, a four dimensional array with indices in
  the order x, y, z and block;
- `meta`: metadata of the field, also a named tuple with:

    - `description`: explanation of what the field is;
    - `dim`: the dimension of the field (if applicable) in SI units.

Tracers data
------------

Tracer data (position, mass, composition...) are accessible through
`Step.tracers` using the property name as key.  They are organized by block.
For example, the masses of tracers in the first block is obtained with
`sdat.snaps[-1].tracers["Mass"][0]`. This is a one dimensional array containing
the mass of each tracers. Their positions can be recovered through the `"x"`,
`"y"` and `"z"` items.
