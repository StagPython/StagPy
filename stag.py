#!/bin/python
import struct
import numpy as np


class ReadStagyyData:

    def __init__(self, fpath, fname, par_type, ti_fn):
        self.fpath = fpath
        self.fname = fname
        self.par_type = par_type
        self.ti_fn = ti_fn
        self.file_format = 'l'

        number_string = '{:05d}'.format(ti_fn)

        # name of the file to read
        self.fullname = fpath + fname + '_' + par_type + number_string

        if par_type in ('t', 'eta', 'rho', 'str', 'age'):
            self.nval = 1
        elif par_type == 'vp':
            self.nval = 4

        with open(self.fullname, 'rb') as self.fid:
            self._catch_header()
            self._readfile()

    def _readbin(self, fmt, nwords=1, nbytes=4):
        """Read n words of n bytes with fmt format.
        Return a tuple of elements if more
        than one element."""

        elts = struct.unpack(fmt*nwords, self.fid.read(nwords*nbytes))
        if len(elts) == 1:
            elts = elts[0]
        return elts

    def _catch_header(self):
        """read header of binary file"""

        self.nmagic = self._readbin('i')  # Version

        # check nb components
        if (self.nmagic < 100 and self.nval > 1) \
            or (self.nmagic > 300 and self.nval == 1):
            raise ValueError('wrong number of components in field')

        # extra ghost point in horizontal direction
        self.xyp = int((self.nmagic % 100) >= 9 and self.nval == 4)

        # total number of values in the...
        self.nthtot = self._readbin('i')  # latitude direction
        self.nphtot = self._readbin('i')  # longitude direction
        self.nrtot = self._readbin('i')   # radius direction

        # number of blocks, 2 for yinyang
        self.nblocks = self._readbin('i')

        # Aspect ratio
        self.aspect = self._readbin('f', 2)
        self.aspect = np.array(self.aspect)

        # Number of parallel subdomains
        # in the th,ph,r and b directions
        self.nnth = self._readbin('i')
        self.nnph = self._readbin('i')
        self.nnr = self._readbin('i')
        self.nnb = self._readbin('i')

        self.nr2 = self.nrtot * 2 + 1
        self.rg = self._readbin('f', self.nr2)  # r-coordinates

        self.rcmb = self._readbin('f')  # radius of the cmb
        self.ti_step = self._readbin('i')
        self.ti_ad = self._readbin('f')
        self.erupta_total = self._readbin('f')
        self.botT_val = self._readbin('f')

        self.th_coord = self._readbin('f', self.nthtot)  # th-coordinates
        self.ph_coord = self._readbin('f', self.nphtot)  # ph-coordinates
        self.r_coord = self._readbin('f', self.nrtot)  # r-coordinates

    def _readfile(self):
        """read scalar/vector fields"""

        # compute nth, nph, nr and nb PER CPU
        nth = self.nthtot / self.nnth
        nph = self.nphtot / self.nnph
        nr = self.nrtot / self.nnr
        nb = self.nblocks / self.nnb
        # the number of values per 'read' block
        npi = (nth + self.xyp) * (nph + self.xyp) * nr * nb * self.nval

        if self.nval > 1:
            self.scalefac = self._readbin('f')
        else:
            self.scalefac = 1

        dim_fields = (self.nblocks, self.nrtot,
                      self.nphtot + self.xyp, self.nthtot + self.xyp)

        flds = []
        for i in range(self.nval):
            flds.append(np.zeros(dim_fields))

        # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        fileContent = self._readbin('f', npi)
                        data_CPU = np.array(fileContent) * self.scalefac

                        # Create a 3D matrix from these data
                        data_CPU_3D = data_CPU.reshape((nb, nr,
                            nph + self.xyp, nth + self.xyp, self.nval))

                        # Add local 3D matrix to global matrix
                        sth = ithc * nth
                        eth = ithc * nth + nth + self.xyp
                        sph = iphc * nph
                        eph = iphc * nph + nph + self.xyp
                        sr = irc * nr
                        er = irc * nr + nr
                        snb = ibc * nb
                        enb = ibc * nb + nb

                        for idx, fld in enumerate(flds):
                            fld[snb:enb, sr:er, sph:eph, sth:eth] = \
                                    data_CPU_3D[:, :, :, :, idx]

        self.fields = []
        for fld in flds:
            self.fields.append(fld[0, :, :, :])
