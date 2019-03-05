Plotting a scalar diagnostic as function of control parameter
=============================================================

Stagpy is suitable for scripting using StagyyData. Suppose you have a group 
of directories, each for a given set of parameters,
and you want to plot the results of all cases on the same figure, compute
statistics etc. The :class:`~stagpy.stagyydata.StagyyData` class comes in handy
for that. The following script can be used to make a loglog plot of the Nusselt
number as function of the Rayleigh number using all directories stored where
the script is executed::

  #!/usr/bin/env python3
  """Nu=f(Ra) from a set of stagyy results in different directories"""

  import matplotlib.pyplot as plt
  import numpy as np
  from stagpy import stagyydata
  from pathlib import Path

  ran = []
  nun = []

  pwd = Path('Examples')
  for rep in pwd.glob('ra-*'):
      print('In directory ', rep)
      sdat = stagyydata.StagyyData(rep)
      # get the value of the Rayleigh number
      ran.append(sdat.par['refstate']['ra0'])
      # get the last value of the Nusselt number
      nun.append(sdat.steps.last.timeinfo['Nutop'])

  ran = np.array(ran)
  nun = np.array(nun)

  # sort by Ra#
  indexes = ran.argsort()

  fig = plt.figure()
  plt.loglog(ran[indexes], nun[indexes], 'o--')
  plt.xlabel(r'Rayleigh number')
  plt.ylabel(r'Nusselt number')
  plt.savefig('Ra-Nu.pdf')
  plt.close(fig)

Note that this particular example is only relevant if the solutions
have all reached a steady-state. In the case where the solution is
only in statistical steady state, a time average is more relevant. It
can be computed using the whole sdat.tseries table in each directory.
