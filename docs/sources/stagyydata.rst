The StagyyData class
====================

The :class:`~stagpy.stagyydata.StagyyData` class is a generic lazy accessor to
StagYY output data (HDF5 or "legacy" format) you can use in your own Python
scripts. This section assumes the :class:`~stagpy.stagyydata.StagyyData`
instance is called ``sdat``. You can create such an instance like this::

    from stagpy import stagyydata
    sdat = stagyydata.StagyyData('path/to/run/')

where ``path/to/run/`` is the path towards the directory containing your run
(where the ``par`` file is). This path can be absolute or relative to the
current working directory. It can be a regular string, or a
:class:`pathlib.Path` object.

Snapshots and time steps
------------------------

A StagYY run is a succession of time steps with information such as the mean
temperature of the domain outputted at each time step. Now and then, radial
profiles and complete fields are saved, constituting a snapshot.

A :class:`~stagpy.stagyydata.StagyyData` instance has two attributes to access
time steps and snapshots in a consistent way:
:attr:`~stagpy.stagyydata.StagyyData.steps` and
:attr:`~stagpy.stagyydata.StagyyData.snaps`. Accessing the ``n``-th time step
or the ``m``-th snapshot is done using the item access notation (square
brackets)::

    sdat.steps[n]
    sdat.snaps[m]

These two expressions each return a :class:`~stagpy._step.Step` instance.
Moreover, if the ``m``-th snapshot was done at the ``n``-th step, both
expressions return the same :class:`~stagpy._step.Step` instance. In
other words, if for example the 100th snapshot was made at the 1000th step,
``sdat.steps[1000] is sdat.snaps[100]`` is true.  The correspondence between
time steps and snapshots is deduced from available binary files.

Negative indices are allowed, ``sdat.steps[-1]`` being the last time step
(inferred from temporal series information) and ``sdat.snaps[-1]`` being the
last snapshot (inferred from available binary files).

Iterators and filters
---------------------

:attr:`~stagpy.stagyydata.StagyyData.steps` and
:attr:`~stagpy.stagyydata.StagyyData.snaps` accessors accept slices.
``sdat.steps[a:b:c]`` returns a :class:`~stagpy.stagyydata._StepsView`
instance. Iterating through this object is similar to iterating through the
generator ``(sdat.steps[n] for n in range(a, b, c))`` (negative indices are
also properly taken care of).  For example, the following code process every
even snapshot::

    for step in sdat.snaps[::2]:
        do_something(step)

:class:`~stagpy.stagyydata._StepsView` instances offer the possibility to
filter out the steps objects with the
:meth:`~stagpy.stagyydata._StepsView.filter` method (see its API reference
documentation for a list of available filters). For example, the following loop
process every timestep from the 512-th that has temperature and viscosity field
data available::

    for step in sdat.steps[512:].filter(fields=['T', 'eta']):
        do_something(step)

As a convenience, attempting to iterate through ``sdat.steps`` and
``sdat.snaps`` is equivalent to iterate through ``sdat.steps[:]`` and
``sdat.snaps[:]``. Similarly, calling ``sdat.steps.filter()`` and
``sdat.snaps.filter()`` is a shortcut for ``sdat.steps[:].filter()``
and ``sdat.snaps[:].filter()``.

Parameters file
---------------

Parameters set in the ``par`` file are accessible through the
:attr:`~stagpy.stagyydata.StagyyData.par` attribute of a
:class:`~stagpy.stagyydata.StagyyData` instance.  ``sdat.par`` is organized as
a dictionary of dictionaries.  For example, to access the Rayleigh number from
the ``refstate`` section of the par file, one can use
``sdat.par['refstate']['ra0']``. Parameters that are not set in the par file
are given a default value according to the par file ``~/.config/stagpy/par``.

Radial profiles
---------------

Radial profile data are accessible trough the :attr:`~stagpy._step.Step.rprofs`
attribute of a :class:`~stagpy._step.Step` instance.  This attribute implements
getitem to access radial profiles.  Keys are the names of available
variables (such as e.g. ``'Tmean'`` and ``'vzabs'``).  Items are named tuples
with three fields:

- :data:`values`: the profile itself;
- :data:`rad`: the radial position at which the profile is evaluated;
- :data:`meta`: metadata of the profile, also a named tuple with:

    - :data:`description`: explanation of what the profile is;
    - :data:`kind`: the category of profile;
    - :data:`dim`: the dimension of the profile (if applicable) in SI units.

The list of available variables can be obtained by running ``% stagpy var``.

For example, ``sdat.steps[1000].rprofs['Tmean']`` is the temperature profile of
the 1000th timestep.

Time series
-----------

Temporal data are accessible through the
:attr:`~stagpy.stagyydata.StagyyData.tseries` attribute of a
:class:`~stagpy.stagyydata.StagyyData` instance. This attribute implements
getitem to access time series.  Keys are the names of available variables
(such as e.g. ``'Tmean'`` and ``'ftop'``).  Items are named tuples with
three fields:

- :data:`values`: the series itself;
- :data:`time`: the times at which the series is evaluated;
- :data:`meta`: metadata of the series, also a named tuple with:

    - :data:`description`: explanation of what the series is;
    - :data:`kind`: the category of series;
    - :data:`dim`: the dimension of the series (if applicable) in SI units.

The list of available variables can be obtained by running ``% stagpy var``.

The time series data at a given time step can be accessed from
:attr:`Step.timeinfo <stagpy._step.Step.timeinfo>`.  For example,
``sdat.steps[1000].timeinfo`` is equivalent to ``sdat.tseries.at_step(1000)``.
Both are :class:`pandas.Series` indexed by the available variables.


Geometry
--------

Geometry information are read from binary files.  :attr:`Step.geom
<stagpy._step.Step.geom>` has various attributes defining the geometry of
the problem.

:attr:`~stagpy._step._Geometry.cartesian`,
:attr:`~stagpy._step._Geometry.curvilinear`,
:attr:`~stagpy._step._Geometry.cylindrical`,
:attr:`~stagpy._step._Geometry.spherical` and
:attr:`~stagpy._step._Geometry.yinyang` booleans define the shape of the domain
(:attr:`~stagpy._step._Geometry.curvilinear` being the opposite of
:attr:`~stagpy._step._Geometry.cartesian`, ``True`` if
:attr:`~stagpy._step._Geometry.cylindrical` or
:attr:`~stagpy._step._Geometry.spherical` is ``True``).

:attr:`~stagpy._step._Geometry.twod_xz`,
:attr:`~stagpy._step._Geometry.twod_yz`, :attr:`~stagpy._step._Geometry.twod`
and :attr:`~stagpy._step._Geometry.threed` booleans indicate the number of
spatial dimensions in the simulation. Note that fields are always four
dimensional arrays (spatial + blocks) regardless of the actual dimension of the
domain.

:attr:`~stagpy._step._Geometry.nxtot`, :attr:`~stagpy._step._Geometry.nytot`,
:attr:`~stagpy._step._Geometry.nztot`, :attr:`~stagpy._step._Geometry.nbtot`,
:attr:`~stagpy._step._Geometry.nttot`, :attr:`~stagpy._step._Geometry.nptot`
and :attr:`~stagpy._step._Geometry.nrtot` are the total number of points in the
various spatial directions. Note that :attr:`~stagpy._step._Geometry.nttot`,
:attr:`~stagpy._step._Geometry.nptot` and :attr:`~stagpy._step._Geometry.nrtot`
are the same as :attr:`~stagpy._step._Geometry.nxtot`,
:attr:`~stagpy._step._Geometry.nytot` and :attr:`~stagpy._step._Geometry.nztot`
regardless of whether the geometry is cartesian or curvilinear.

:attr:`~stagpy._step._Geometry.x_centers`,
:attr:`~stagpy._step._Geometry.y_centers`, and
:attr:`~stagpy._step._Geometry.z_centers` as well as
:attr:`~stagpy._step._Geometry.t_centers`,
:attr:`~stagpy._step._Geometry.p_centers`, and
:attr:`~stagpy._step._Geometry.r_centers` are the coordinates of cell centers
in the three directions.  As for the total number of points, they are the same
regardless of the actual geometry.

Similarly to ``*_centers`` attributes, :attr:`~stagpy._step._Geometry.x_walls`,
:attr:`~stagpy._step._Geometry.y_walls`, and
:attr:`~stagpy._step._Geometry.z_walls` as well as
:attr:`~stagpy._step._Geometry.t_walls`,
:attr:`~stagpy._step._Geometry.p_walls`, and
:attr:`~stagpy._step._Geometry.r_walls` are the coordinates of cell walls in
the three directions.

Scalar and vector fields
------------------------

Vector and scalar fields are accessible through :attr:`Step.fields
<stagpy._step.Step.fields>` using their name as key. For example, the
temperature field of the 100th snapshot is obtained with
``sdat.snaps[100].fields['T']``.  Valid names of fields can be obtained by
running ``% stagpy var``. Items are named tuples with two elements:

- :data:`values`: the field itself, a four dimensional array with indices in
  the order x, y, z and block;
- :data:`meta`: metadata of the field, also a named tuple with:

    - :data:`description`: explanation of what the field is;
    - :data:`dim`: the dimension of the field (if applicable) in SI units.

Tracers data
------------

Tracer data (position, mass, composition...) are accessible through
:attr:`Step.tracers<stagpy._step.Step.tracers>` using the
property name as key.  They are organized by block.  For example,
the masses of tracers in the first block is obtained with
``sdat.snaps[-1].tracers['Mass'][0]``. This is a one dimensional
array containing the mass of each tracers. Their positions can be
recovered through the ``'x'``, ``'y'`` and ``'z'`` items.

