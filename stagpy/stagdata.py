"""define StagyyData"""

import struct
import numpy as np
from scipy import integrate
from . import constants, misc


class BinData:

    """reads StagYY binary data and processes them"""

    def __init__(self, args, var, timestep):
        """read the necessary binary file

        after init, the StagyyData object is ready
        for processing
        """
        self.args = args
        self.var = var
        self.par_type = constants.FIELD_VAR_LIST[var].par
        self.geom = args.geometry
        self.file_format = 'l'
        self.step = timestep

        # name of the file to read
        self.fullname = misc.path_fmt(args, self.par_type).format(timestep)
        self.nval = 4 if self.par_type == 'vp' else 1

        with open(self.fullname, 'rb') as self._fid:
            self._catch_header()
            self._readfile()

    def _readbin(self, fmt='i', nwords=1, nbytes=4):
        """read n words of n bytes with fmt format

        Return a tuple of elements if more than one element.
        Default: read 1 word of 4 bytes formatted as an integer.
        """
        elts = struct.unpack(fmt*nwords, self._fid.read(nwords*nbytes))
        if len(elts) == 1:
            elts = elts[0]
        return elts

    def _catch_header(self):
        """reads header of binary file"""
        self.nmagic = self._readbin()  # Version

        # check nb components
        if (self.nmagic < 100 and self.nval > 1) \
                or (self.nmagic > 300 and self.nval == 1):
            raise ValueError('wrong number of components in field')

        # extra ghost point in horizontal direction
        self.xyp = int((self.nmagic % 100) >= 9 and self.nval == 4)

        # total number of values in
        # latitude, longitude and radius directions
        self.nthtot, self.nphtot, self.nrtot = self._readbin(nwords=3)

        # number of blocks, 2 for yinyang
        self.nblocks = self._readbin()

        # Aspect ratio
        self.aspect = self._readbin('f', 2)
        self.aspect = np.array(self.aspect)

        # Number of parallel subdomains in the th,ph,r and b directions
        self.nnth, self.nnph, self.nnr = self._readbin(nwords=3)
        self.nnb = self._readbin()

        self.nr2 = self.nrtot * 2 + 1
        self.rgeom = self._readbin('f', self.nr2)  # r-coordinates

        self.rcmb = self._readbin('f')  # radius of the cmb
        self.ti_step = self._readbin()
        self.ti_ad = self._readbin('f')
        self.erupta_total = self._readbin('f')
        self.bot_temp = self._readbin('f')

        self.th_coord = self._readbin('f', self.nthtot)  # th-coordinates
        self.ph_coord = self._readbin('f', self.nphtot)  # ph-coordinates
        self.r_coord = self._readbin('f', self.nrtot)  # r-coordinates

    def _readfile(self):
        """read scalar/vector fields"""
        # compute nth, nph, nr and nb PER CPU
        nth = self.nthtot // self.nnth
        nph = self.nphtot // self.nnph
        nrd = self.nrtot // self.nnr
        nbk = self.nblocks // self.nnb
        # the number of values per 'read' block
        npi = (nth + self.xyp) * (nph + self.xyp) * nrd * nbk * self.nval

        if self.nval > 1:
            self.scalefac = self._readbin('f')
        else:
            self.scalefac = 1

        dim_fields = (self.nblocks, self.nrtot,
                      self.nphtot + self.xyp, self.nthtot + self.xyp)

        flds = []
        for _ in range(self.nval):
            flds.append(np.zeros(dim_fields))

        # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        file_content = self._readbin('f', npi)
                        data_cpu = np.array(file_content) * self.scalefac

                        # Create a 3D matrix from these data
                        data_cpu_3d = data_cpu.reshape(
                            (nbk, nrd, nph + self.xyp,
                             nth + self.xyp, self.nval))

                        # Add local 3D matrix to global matrix
                        sth = ithc * nth
                        eth = ithc * nth + nth + self.xyp
                        sph = iphc * nph
                        eph = iphc * nph + nph + self.xyp
                        srd = irc * nrd
                        erd = irc * nrd + nrd
                        snb = ibc * nbk
                        enb = ibc * nbk + nbk

                        for idx, fld in enumerate(flds):
                            fld[snb:enb, srd:erd, sph:eph, sth:eth] = \
                                    data_cpu_3d[:, :, :, :, idx]

        self.fields = {}
        fld_names = ['u', 'v', 'w', 'p'] if self.par_type == 'vp' \
                else [self.var]
        for fld_name, fld in zip(fld_names, flds):
            self.fields[fld_name] = fld[0, :, :, :]

    def calc_stream(self):
        """computes and returns the stream function

        only make sense with vp fields
        """
        # should add test if vp fields or not
        vphi = self.fields['vy'][:, :, 0]
        vph2 = -0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
        v_r = self.fields['vz'][:, :, 0]
        n_r, nph = np.shape(v_r)
        stream = np.zeros(np.shape(vphi))
        # integrate first on phi
        stream[0, 1:nph - 1] = self.rcmb * \
            integrate.cumtrapz(v_r[0, 0:nph - 1], self.ph_coord)
        stream[0, 0] = 0
        # use r coordinates where vphi is defined
        rcoord = self.rcmb + np.array(
            self.rgeom[0:np.shape(self.rgeom)[0] - 1:2])
        for iph in range(0, np.shape(vph2)[1] - 1):
            stream[1:n_r, iph] = stream[0, iph] + \
                integrate.cumtrapz(vph2[:, iph], rcoord)  # integrate on r
        stream = stream - np.mean(stream[n_r / 2, :])
        # remove some typical value. Would be better to compute the golbal average
        # taking into account variable grid spacing
        return stream

