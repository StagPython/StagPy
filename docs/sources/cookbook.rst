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

Profiles
~~~~~~
Profiles are accessed using the rprof command::

  % stagpy rprof -o t -s 4:6

In this example, mean temperature profiles of snapshot 4 and 5 are
plotted on the same graph. The average is also computed on another
graph.

::

   stagpy rprof -o t -t 500:

plots all mean temperature profiles that have been saved starting from
time-step 500.

::

   stagpy rprof -o T

plots the last min, mean and max temperature profiles.

::

   stagpy rprof -o g

plots the last snapshot of the grid spacing profile.



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
      # get the last value (-1) of the Nusselt number (column 2)
      nun.append(sdat.tseries[-1, 2])

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

  fig = plt.figure()

  pwd = Path('.')
  for rep in pwd.glob('ra-*'):
      print('In directory ', rep)
      sdat = stagyydata.StagyyData(rep.name)
      # get the value of the Rayleigh number
      ra0 = sdat.par['refstate']['ra0']
      # get the time vector
      time = sdat.tseries[:, 1]
      # get the vrms vector
      vrms = sdat.tseries[:, 8]
      # plot
      plt.plot(time, vrms, label=r'$Ra=%1.0e$' % ra0)

  plt.legend()
  plt.xlabel(r'Time')
  plt.ylabel(r'RMS velocity')
  plt.savefig('time-vrms.pdf')
  plt.close(fig)




