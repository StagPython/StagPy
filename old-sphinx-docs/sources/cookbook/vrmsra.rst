Plotting a scalar diagnostic as function of time for several parameter sets
===========================================================================

Instead of plotting just the last value of a diagnostic, or its average, you
may want to plot its evolution of time for different values of the control
parameters. Suppose again that several directories named ra-* are present in
your working directory. The following script will plot the RMS velocity as
function of time for all these directories::

    #!/usr/bin/env python3
    """vrms(t) from a set of StagYY runs."""

    from pathlib import Path
    from stagpy.stagyydata import StagyyData
    import matplotlib.pyplot as plt
    import numpy as np

    pwd = Path('Examples/')
    for folder in pwd.glob('ra-*'):
        print('In directory', folder)
        sdat = StagyyData(folder)
        # Reference Rayleigh number as a power of ten
        ra0_log10 = int(np.log10(sdat.par['refstate']['ra0']))
        # vrms time series object
        vrms = sdat.tseries['vrms']
        # plot time vs vrms values
        plt.plot(vrms.time, vrms.values, label=f'$Ra=10^{{{ra0_log10}}}$')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('RMS velocity')
    plt.savefig('time-vrms.pdf')
