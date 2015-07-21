"""post-processing functions"""
import numpy as np
from scipy import integrate


def calc_stream(stagdata):
    """computes and returns the stream function for
    a StagData object containing vp fields"""

    vphi = stagdata.fields[1][:, :, 0]
    vph2 = -0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    vr = stagdata.fields[2][:, :, 0]
    nr, nph = np.shape(vr)
    stream = np.zeros(np.shape(vphi))
    # integrate first on phi
    stream[0, 1:nph - 1] = stagdata.rcmb * \
        integrate.cumtrapz(vr[0, 0:nph - 1], stagdata.ph_coord)
    stream[0, 0] = 0
    # use r coordinates where vphi is defined
    rcoord = stagdata.rcmb + np.array(
        stagdata.rg[0:np.shape(stagdata.rg)[0] - 1:2])
    for iph in range(0, np.shape(vph2)[1] - 1):
        stream[1:nr, iph] = stream[0, iph] + \
            integrate.cumtrapz(vph2[:, iph], rcoord)  # integrate on r
    stream = stream - np.mean(stream[nr / 2, :])
    # remove some typical value. Would be better to compute the golbal average
    # taking into account variable grid spacing
    return stream
