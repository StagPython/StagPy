Getting started
===============

StagPy is primarily designed as a command line tool. This section contains
instructions to use StagPy *via* the command line.

Read the :doc:`installation instructions <install>` first in order to have
StagPy available on your system. The rest of this documentation assumes that
you have installed StagPy and that you can call it from the command line with
the command ``stagpy``.

Subcommands
-----------

The various processing capabilities of StagPy are organized in subcommands.
This means a minimal call to StagPy is as follow::

    stagpy <subcommand>

``<subcommand>`` can be one of the following:

* ``field``: compute and/or plots scalar fields such as temperature or stream
  function;
* ``rprof``: compute and/or plots radial profiles;
* ``time``: compute and/or plots time series;
* ``plates``: plate analysis;
* ``var``: display a list of available variables;
* ``version``: display the installed version of StagPy;
* ``config``: configuration handling.

You can run ``stagpy --help`` (or ``stagpy -h``) to display a help message
describing those subcommands. You can also run ``stagpy <subcommand> --help``
to have some help on the available options for one particular sub command.

What StagPy does
----------------

StagPy looks for a StagYY ``par`` file in the current directory. It then reads
the value of the ``output_file_stem`` entry to determine the location and name
of the StagYY output files (set to ``test`` if no ``par`` file can be found).
You can change the directory in which StagYY looks for a ``par`` file by two
different ways:

* you can change the default behavior in a global way by editing the config
  file (``stagpy config --edit``) and change the ``core.path`` variable;
* or you can change the path only for the current run with the ``-p`` option.

Options
-------

The time step option ``-s`` allows you to specify a range of time steps in a
way which mimic the slicing syntax: ``begin:end:gap`` (both ends included). If
the first step is not specified, it is set to ``0``. If the final step is not
specified, all available time steps are processed. Here are some examples:

* ``-s 100:350`` will process every time steps between 100 and 350;
* ``-s 201:206:2`` will process time steps 201, 203 and 205;
* ``-s 201:205:2`` same as previous;
* ``-s 5682:`` will process every time steps from the 5682nd to the last one;
* ``-s :453`` will process every time steps from the 0th to the 453rd one;
* ``-s ::2`` will process every even time steps.

By default, the temperature, pressure and stream function fields are plotted.
You can change this with the ``-o`` option (e.g. ``./main.py field -o ps`` to
plot only the pressure and stream function fields).

