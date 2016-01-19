"""post-processing functions"""
import numpy as np
from scipy import integrate


def calc_stream(stagdata):
    """computes and returns the stream function

    need a StagData object containing vp fields
    """
    vphi = stagdata.fields[1][:, :, 0]
    vph2 = -0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    v_r = stagdata.fields[2][:, :, 0]
    n_r, nph = np.shape(v_r)
    stream = np.zeros(np.shape(vphi))
    # integrate first on phi
    stream[0, 1:nph - 1] = stagdata.rcmb * \
        integrate.cumtrapz(v_r[0, 0:nph - 1], stagdata.ph_coord)
    stream[0, 0] = 0
    # use r coordinates where vphi is defined
    rcoord = stagdata.rcmb + np.array(
        stagdata.rgeom[0:np.shape(stagdata.rgeom)[0] - 1:2])
    for iph in range(0, np.shape(vph2)[1] - 1):
        stream[1:n_r, iph] = stream[0, iph] + \
            integrate.cumtrapz(vph2[:, iph], rcoord)  # integrate on r
    stream = stream - np.mean(stream[n_r / 2, :])
    # remove some typical value. Would be better to compute the golbal average
    # taking into account variable grid spacing
    return stream
