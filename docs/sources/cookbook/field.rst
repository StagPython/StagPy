Snapshots using command line
============================

For the examples here, simply copy and paste the command line in your
shell, working in the directory where the StagYY par file is located. 
You can also use the examples on the data available in the Examples
directory. 

The next use of stagpy is to create images of snapshots of the
different fields.

::

   % stagpy field -o T

will plot the last snapshot of the temperature field.

::

    % stagpy field -s : -o T-v3

will plot all the snapshots of the temperature and the vertical/radial velocity
on separate figures.

::

    % stagpy field -s 3:8:2 -o T,stream

will plot the temperature field with isocontours of the stream function from
the third to the eighth snapshot, every two snapshots.

::

    % stagpy field -o T -s 1,5 --vmin=0.8 --vmax=1.0
   
will plot the temperature field from the first and the fifth snapshot while
keeping the range of the colorbar fixed between 0.8 and 1.

List of variables to plot
-------------------------

The list of fields specified with the ``-o`` (or ``--plot``) option follows the
same rules as for the ``rprof`` and ``time`` subcommands.  Namely,
``,``-separated variables are on the same subplots; ``.``-separated variables
are on the same figure but different subplots; ``-``-separated variables are
on different figures.

Note that only two fields can be on the same subplot, the first field is a
color map and the second field can be either:

- a scalar field, isocontours are added to the plot;
- a vector field (e.g. ``v`` for the ``(v1, v2, v3)`` vector), arrows are added
  to the plot.

For example, ``-o=T,v3`` asks for a temperature map with isocontour of the
vertical velocity, while ``-o=T,v`` asks for a temperature map with velocity
vectors on top of it.

If you ask for more than two fields on the same subplot, extra fields are
ignored.  ``-o=T,stream,v`` is therefore equivalent to ``-o=T,stream``.
