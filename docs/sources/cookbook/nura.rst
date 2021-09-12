Plotting a scalar diagnostic as function of control parameter
=============================================================

StagPy is suitable for scripting using StagyyData. Suppose you have a group
of directories, each for a given set of parameters,
and you want to plot the results of all cases on the same figure, compute
statistics etc. The :class:`~stagpy.stagyydata.StagyyData` class comes in handy
for that. The following script can be used to make a loglog plot of the Nusselt
number as function of the Rayleigh number using all directories stored where
the script is executed::

    #!/usr/bin/env python3
    """Nu=f(Ra) from a set of stagyy results in different directories"""

    from pathlib import Path
    from stagpy.stagyydata import StagyyData
    import matplotlib.pyplot as plt
    import numpy as np

    ran = []
    nun = []

    pwd = Path('Examples')
    for folder in pwd.glob('ra-*'):
        print('In directory', folder)
        sdat = StagyyData(folder)
        # get the value of the Rayleigh number
        ran.append(sdat.par['refstate']['ra0'])
        # get the last value of the Nusselt number
        nun.append(sdat.tseries['Nutop'].values[-1])

    ran = np.array(ran)
    nun = np.array(nun)

    # sort by Ra#
    indexes = ran.argsort()

    plt.loglog(ran[indexes], nun[indexes], 'o--')
    plt.xlabel('Rayleigh number')
    plt.ylabel('Nusselt number')
    plt.savefig('Ra-Nu.pdf')

Note that this particular example is only relevant if the solutions
have all reached a steady-state. In the case where the solution is
only in statistical steady state, a time average is more relevant. It
can be computed using :func:`~stagpy.time_series.compstat`.
