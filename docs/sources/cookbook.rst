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

will give you by default on figure with two subplots. The first subplot
contains the time series of the Nusselt number at the top and bottom
boundaries, with a check of the energy balance. The corresponding line should
be zero at all times for a perfect balance. The second subplot contains the
time series of the mean temperature.

::

   % stagpy time --tstart 0.02 --tend 0.03

will give you the same plots but starting at time 0.02 and ending at
time 0.03.

::

    % stagpy time -o vrms_Tmin,Tmean,Tmax.dTdt

creates two figures. The first one contains the time series of the rms
velocity. The second one contains two subplots, the first one with the time
series of the minimal, maximal and average temperature; the second one with the
time derivative of the mean temperature. The variable names can be found by
running the ``% stagpy var`` command. The variables you want on the same
subplot are separated by commas ``,``, the variables you want on different
subplots are separated by dots ``.``, and the variables you want on different
figures are separated by underscores ``_``.

::

   % stagpy time +compstat --tstart 0.02

will create a file containing the average and standard deviation of the time
series variables from t=0.02 to the end. This is useful when your system has
reached a statistical steady state.


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

Profiles
~~~~~~
Profiles are accessed using the rprof command::

    % stagpy rprof -s 4:6

In this example, mean temperature profiles of snapshot 4 and 5 are
plotted on two graph.

::

    % stagpy rprof -s 4:6 +a

plots the average radial profiles of snapshots 4 and 5.

::

    % stagpy rprof -o Tmin,Tmean,Tmax.vzabs,vhrms -t 500:

plots all temperature and velocity profiles that have been saved starting from
time-step 500. The list of variables you want follow the same logic as time
series variables.

::

    % stagpy rprof +g -o

plots grid spacing profile for the last snapshot available. The ``-o`` flag
turns off output of other radial profiles (``Tmean`` by default).



Scripts using StagyyData
--------------------

Plotting a scalar diagnostic as function of control parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose you have a group of directories, each for a given set of
parameters, and you want to plot the results of all cases on the same
figure, compute statistics etc. The StagyyData is specially designed
for that. The following script can be used to make a loglog plot of
the Nusselt number as function of the Rayleigh number using
all directories stored where the script is executed::

  #!/usr/bin/env python3
  """Nu=f(Ra) from a set of stagyy results in different directories"""

  import matplotlib.pyplot as plt
  from stagpy import stagyydata
  from pathlib import Path

  ran =[]
  nun = []

  pwd = Path('.')
  for rep in pwd.glob('ra-*'):
      print('In directory ', rep)
      sdat = stagyydata.StagyyData(rep.name)
      # get the value of the Rayleigh number
      ran.append(sdat.par['refstate']['ra0'])
      # get the last value of the Nusselt number
      nun.append(sdat.steps.last.timeinfo['Nutop'])

  fig = plt.figure()
  plt.loglog(ran, nun, 'o--')
  plt.xlabel(r'Rayleigh number')
  plt.ylabel(r'Nusselt number')
  plt.savefig('Ra-Nu.pdf')
  plt.close(fig)

Note that this particular example is only relevant if the solutions
have all reached a steady-state. In the case where the solution is
only in statistical steady state, a time average is more relevant. It
can be computed using the whole sdat.tseries table in each directory.

Plotting a scalar diagnostic as function of time for several parameter sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of plotting just the last value of a diagnostic, or its
average, you may want to plot its evolution of time for different
values of the control parameters. Suppose again that several
directories named ra-* are present in your working directory. The
following script will plot the RMS velocity (column 8 of the tseries
table) as function of time for all these directories::

  #!/usr/bin/env python3
  """Nu=f(Ra) from a set of stagyy results in different directories"""

  import matplotlib.pyplot as plt
  from stagpy import stagyydata
  from pathlib import Path
  from numpy import log10

  fig = plt.figure()

  pwd = Path('.')
  for rep in pwd.glob('ra-*'):
      print('In directory ', rep)
      sdat = stagyydata.StagyyData(rep.name)
      # get the value of the Rayleigh number
      ra0 = sdat.par['refstate']['ra0']
      # get the time vector
      time = sdat.tseries['t']
      # get the vrms vector
      vrms = sdat.tseries['vrms']
      # plot
      plt.plot(time, vrms, label=r'$Ra=10^{%1d}$' % log10(ra0))

  plt.legend()
  plt.xlabel(r'Time')
  plt.ylabel(r'RMS velocity')
  plt.savefig('time-vrms.pdf')
  plt.close(fig)

