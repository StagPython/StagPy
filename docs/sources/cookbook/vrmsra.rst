Plotting a scalar diagnostic as function of time for several parameter sets
===========================================================================

Instead of plotting just the last value of a diagnostic, or its average, you
may want to plot its evolution of time for different values of the control
parameters. Suppose again that several directories named ra-* are present in
your working directory. The following script will plot the RMS velocity as
function of time for all these directories::

  #!/usr/bin/env python3
  """Nu=f(Ra) from a set of stagyy results in different directories"""

  import matplotlib.pyplot as plt
  from stagpy import stagyydata
  from pathlib import Path
  from numpy import log10

  fig = plt.figure()

  pwd = Path('Examples/')
  for rep in pwd.glob('ra-*'):
      print('In directory ', rep)
      sdat = stagyydata.StagyyData(rep)
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
