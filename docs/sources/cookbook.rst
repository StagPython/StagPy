Cookbook
=======
You will find here some examples of use that you can try on the data
available in the Examples directory.

Simple command lines
------------------
For the examples here, simply copy and paste the command line in your
shell, working in the directory where the StagYY par file is located.

Time series
~~~~~~~~~

The command

::

    % stagpy time

will give you two plots with two subplots each of time series. One has the mean
temperature at the bottom and the top and bottom heat fluxes at the
top while the other one shows the mean temperature and the RMS velocity.

::

    % stagpy time +energy

adds a check of the energy balance to the heat flow time series. The
corresponding line should be zero at all times for a perfect balance.

::

   % stagpy time --tstart 0.02 --tend 0.03

will give you the same plots but starting at time 0.02 and ending at
time 0.03.

Snapshots
~~~~~~~~
The next use of stagpy is to create images of snapshots of the
different fields.

::

   % stagpy field -o t

will plot the last snapshot of the temperature field.

::

   % stagpy field -s : -o tw

will plot all the snapshots of t and w on separate figures.

::

   % stagpy field -s 3:8:2 -o tp

will plot temperature and pressure snapshots from the third to the
eighth, every two snapshots.


Scripts using StagyyData
--------------------

