Plotting tracers information
============================

StagPy gives you access to tracers information through StagyyData. Mind that
tracers information is organized by blocks even if your run only has one
block. The following script offers an example::

    from stagpy.stagyydata import StagyyData
    import matplotlib.pyplot as plt
    import numpy as np

    sdat = StagyyData('.')

    time = []
    energy = []
    for snap in sdat.snaps:
        time.append(snap.time)
        energy.append(np.sum(snap.tracers['Mass'][0] *
                             snap.tracers['Temperature'][0]))

    fig, axes = plt.subplots(nrows=2)

    axes[0].plot(time, energy)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Internal energy")

    axes[1].hist(sdat.snaps[-1].tracers['TimeMelted'][0])
    axes[1].set_xlabel("Time melted")
    axes[1].set_ylabel("Number of tracers")

    fig.tight_layout()
    plt.savefig('tracers.pdf')
