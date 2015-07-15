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

    def _readbin(self, fid, fmt, nwords=1, nbytes=4):
        """Read n words of n bytes with fmt format
        from the fid file, update self.byte_offset.
        Return a tuple of elements if more
        than one element."""

        elts = struct.unpack(fmt*nwords, fid.read(nwords*nbytes))
        if len(elts) == 1:
            elts = elts[0]
        self.byte_offset += nwords * nbytes
        return elts

    #==========================================================================
    # READ HEADER
    #==========================================================================
    def catch_header(self):
        fid = open(self.fullname, 'rb')  # open file
        self.byte_offset = 0
        self.nmagic = self._readbin(fid, 'i')  # Version

        # check nb components
        if (self.nmagic < 100 and self.nval > 1) or (self.nmagic > 300 and self.nval == 1):
            raise ValueError('wrong number of components in field')

        nnmagic = self.nmagic % 100
        if nnmagic >= 9 and self.nval == 4:
            self.xyp = 1         # extra ghost point in horizontal direction
        else:
            self.xyp = 0

        # total number of values in the latitude direction
        self.nthtot = self._readbin(fid, 'i')
        # total number of values in the longitude direction
        self.nphtot = self._readbin(fid, 'i')
        # total number of values in the radius direction
        self.nrtot = self._readbin(fid, 'i')
        # of blocks, 2 for yinyan
        self.nblocks = self._readbin(fid, 'i')

        # Aspect ratio
        self.aspect = self._readbin(fid, 'f', 2)
        self.aspect = np.array(self.aspect)

        # Number of parallel subdomains
        # in the th,ph,r and b directions
        self.nnth = self._readbin(fid, 'i')
        self.nnph = self._readbin(fid, 'i')
        self.nnr = self._readbin(fid, 'i')
        self.nnb = self._readbin(fid, 'i')

        self.nr2 = self.nrtot * 2 + 1
        self.rg = self._readbin(fid, 'f', self.nr2)  # r-coordinates

        self.rcmb = self._readbin(fid, 'f')  # radius of the cmb
        self.ti_step = self._readbin(fid, 'i')
        self.ti_ad = self._readbin(fid, 'f')
        self.erupta_total = self._readbin(fid, 'f')
        self.botT_val = self._readbin(fid, 'f')

        self.th_coord = self._readbin(fid, 'f', self.nthtot)  # th-coordinates
        self.ph_coord = self._readbin(fid, 'f', self.nphtot)  # ph-coordinates
        self.r_coord = self._readbin(fid, 'f', self.nrtot)  # r-coordinates

        fid.close()

    #==========================================================================
    # READ SCALAR FILE
    #==========================================================================
    def read_scalar_file(self):
        fid = open(self.fullname, 'rb')  # open file
        fid.seek(self.byte_offset)

        # compute nth, nph, nr and nb PER CPU
        nth = self.nthtot / self.nnth
        nph = self.nphtot / self.nnph
        nr = self.nrtot / self.nnr
        nb = self.nblocks / self.nnb
        # the number of values per 'read' block
        npi = (nth + self.xyp) * (nph + self.xyp) * nr * nb * self.nval

        field_3D = np.zeros(
            (self.nblocks, self.nrtot, self.nphtot, self.nthtot))

        # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        fileContent = self._readbin(fid, 'f', npi)
                        data_CPU = np.array(fileContent)
                        # Create a 3D matrix from these data
                        #data_CPU_3D = data_CPU.reshape((nth,nph,nr,nb))
                        #data_CPU_3D = data_CPU.reshape((nr,nph,nth,nb))
                        data_CPU_3D = data_CPU.reshape((nb, nr, nph, nth))
                        # Add local 3D matrix to global matrix
                        sth = ithc * nth
                        eth = ithc * nth + nth
                        sph = iphc * nph
                        eph = iphc * nph + nph
                        sr = irc * nr
                        er = irc * nr + nr
                        snb = ibc * nb
                        enb = ibc * nb + nb
                        field_3D[
                            snb:enb, sr:er, sph:eph, sth:eth] = data_CPU_3D

        self.field = field_3D[0, :, :, :]
        fid.close()

    #==========================================================================
    # READ VECTOR FILE
    #==========================================================================
    def read_vector_file(self):
        fid = open(self.fullname, 'rb')  # open file
        fid.seek(self.byte_offset)

        # compute nth, nph, nr and nb PER CPU
        nth = self.nthtot / self.nnth
        nph = self.nphtot / self.nnph
        nr = self.nrtot / self.nnr
        nb = self.nblocks / self.nnb
        # the number of values per 'read' block
        npi = (nth + self.xyp) * (nph + self.xyp) * nr * nb * self.nval

        self.scalefac = self._readbin(fid, 'f')

        vx_3D = np.zeros(
            (self.nblocks, self.nrtot, self.nphtot + self.xyp, self.nthtot + self.xyp))
        vy_3D = np.zeros(
            (self.nblocks, self.nrtot, self.nphtot + self.xyp, self.nthtot + self.xyp))
        vz_3D = np.zeros(
            (self.nblocks, self.nrtot, self.nphtot + self.xyp, self.nthtot + self.xyp))
        p_3D = np.zeros(
            (self.nblocks, self.nrtot, self.nphtot + self.xyp, self.nthtot + self.xyp))

        # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        fileContent = self._readbin(fid, 'f', npi)
                        data_CPU = np.array(fileContent)
                        data_CPU = data_CPU * self.scalefac
                        # Create a 3D matrix from these data
                        data_CPU_3D = data_CPU.reshape(
                            (nb, nr, nph + self.xyp, nth + self.xyp, self.nval))
                        # Add local 3D matrix to global matrix
                        sth = ithc * nth
                        eth = ithc * nth + nth + self.xyp
                        sph = iphc * nph
                        eph = iphc * nph + nph + self.xyp
                        sr = irc * nr
                        er = irc * nr + nr
                        snb = ibc * nb
                        enb = ibc * nb + nb
                        vx_3D[snb:enb, sr:er, sph:eph, sth:eth] = data_CPU_3D[
                            :, :, :, :, 0]
                        vy_3D[snb:enb, sr:er, sph:eph, sth:eth] = data_CPU_3D[
                            :, :, :, :, 1]
                        vz_3D[snb:enb, sr:er, sph:eph, sth:eth] = data_CPU_3D[
                            :, :, :, :, 2]
                        p_3D[snb:enb, sr:er, sph:eph, sth:eth] = data_CPU_3D[
                            :, :, :, :, 3]

        self.vx = vx_3D[0, :, :, :]
        self.vy = vy_3D[0, :, :, :]
        self.vz = vz_3D[0, :, :, :]
        self.p = p_3D[0, :, :, :]
        fid.close()
