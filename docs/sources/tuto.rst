Getting started
===============

StagPy is both a command line tool and Python module. This section contains
basic instructions on how to use these two flavors of StagPy.

Read the :doc:`installation instructions <install>` first in order to have
StagPy available on your system. The rest of this documentation assumes that
you have installed StagPy and that you can call it from the command line with
the command ``% stagpy``. Command examples all begin with a ``%`` sign,
representing your command prompt.

Command line tool
-----------------

The various processing capabilities of StagPy are organized in subcommands.
This means a minimal call to StagPy is as follow::

    % stagpy <subcommand>

``<subcommand>`` can be one of the following:

* ``field``: compute and/or plots scalar fields such as temperature or stream
  function;
* ``rprof``: compute and/or plots radial profiles;
* ``time``: compute and/or plots time series;
* ``plates``: plate analysis;
* ``var``: display a list of available variables;
* ``version``: display the installed version of StagPy;
* ``config``: configuration handling.

You can run ``% stagpy --help`` (or ``% stagpy -h``) to display a help message
describing those subcommands. You can also run ``% stagpy <subcommand> --help``
to have some help on the available options for one particular sub command.

A simple example would be::

    % stagpy field -p path/to/run/ -o tp -s 42

This asks StagPy to plot the temperature and pressure fields of snapshot 42
of the run lying in ``./path/to/run``. When not specified, the path defaults to
``./`` (i.e. the current directory) and the snapshot defaults to the last one
available. The command ``% stagpy var`` displays the list of fields available
with the ``-o`` option.

See the :doc:`dedicated section <cli>` for more information on the command line
interface.

Snapshots and time steps
------------------------

StagPy allows you to work seamlessly with time steps and snapshots indices.  A
snapshot index is the number of registered profiles and fields, and a time step
index is the number of atomic iterations performed in StagYY.

The snapshots option ``-s`` allows you to specify a range of snapshots in a way
which mimic the slicing syntax: ``begin:end:gap`` (``end`` excluded).
Similarly, the timesteps option ``-t`` allows you to specify a range of time
steps. For example, if snapshots are taken every 10 timesteps, ``-t 100:1001``
is equivalent to ``-s 10:101``.

If the first step/snapshot is not specified, it is set to ``0``. If the final
step/snapshot is not specified, all available steps/snapshots are processed.
Negative indices are allowed (meaning a counting from the last step/snapshot
available). Here are some examples:

* ``-t 100:350`` will process every time steps between 100 and 349;
* ``-t 201:206:2`` will process time steps 201, 203 and 205;
* ``-t 201:205:2`` will process time steps 201 and 203;
* ``-s -10:`` will process the last ten snapshots;
* ``-s :454`` will process every snapshots from the 0th to the 453rd one;
* ``-s ::2`` will process every even snapshots.

Python interface
----------------

StagPy lets you operate the interface it uses internally to access StagYY
output data. This allows you to write your own scripts to do some specific
processings that aren't implemented in StagPy.

The interface is wrapped in the ``stagpy.stagyydata.StagyyData`` class.
Instantiating and using this class is rather simple::

    from stagpy.stagyydata import StagyyData
    sdat = StagyyData('path/to/run/')

    # absolute vertical velocity profile of last snapshot
    last_v_prof = sdat.snaps[-1].rprof['vzabs']

    # temperature field of the 10000th time step
    # (will be None if no snapshot is available at this timestep)
    temp_field = sdat.steps[10000].fields['t']

    # iterate through snaps 100, 105, 110... up to the last one
    for snap in sdat.snaps[100::5]:
        do_something(snap)

As you can see, the snapshot/time step distinction is automatically taken care
of by ``StagyyData``.

All output data available in the StagYY run is accessible through this
interface. ``StagyyData`` is designed a lazy data accessor. This means output
files are read only when the data they contain is asked for. For example, the
temperature field of the last snapshot isn't read until
``sdat.snaps[-1].fields['t']`` is asked for.

See the :doc:`dedicated section <stagyydata>` for more information on the
``StagyyData`` class.

