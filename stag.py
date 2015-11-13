"""define StagyyData"""

import struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import constants
import misc


class StagyyData(object):

    """reads StagYY binary data and processes them"""

    def __init__(self, args, par_type, timestep):
        """read the necessary binary file

        after init, the StagyyData object is ready
        for processing
        """
        self.args = args
        self.par_type = par_type
        self.geom = args.geometry
        self.file_format = 'l'
        self.step = timestep

        # name of the file to read
        self.fullname = misc.path_fmt(args, par_type).format(timestep)
        if par_type in ('t', 'c', 'eta', 'rho', 'str', 'age'):
            self.nval = 1
        elif par_type == 'vp':
            self.nval = 4

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
        nth = self.nthtot / self.nnth
        nph = self.nphtot / self.nnph
        nrd = self.nrtot / self.nnr
        nbk = self.nblocks / self.nnb
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

        self.fields = []
        for fld in flds:
            self.fields.append(fld[0, :, :, :])

    def plot_scalar(self, var):
        """var: one of the key of constants.varlist"""
        fld = constants.varlist[var].func(self)

        # adding a row at the end to have continuous field
        if self.geom == 'annulus':
            # temp,composition,viscosity,density
            if var in ('t', 'c', 'v', 'd'):
                newline = fld[:, 0, 0]
                fld = np.vstack([fld[:, :, 0].T, newline]).T
            elif var == 'p':
                fld = fld[:, :, 0]
            self.ph_coord = np.append(
                self.ph_coord, self.ph_coord[1] - self.ph_coord[0])

        xmesh, ymesh = np.meshgrid(
            np.array(self.ph_coord), np.array(self.r_coord) + self.rcmb)

        fig, axis = plt.subplots(ncols=1, subplot_kw={'projection': 'polar'})
        if self.geom == 'annulus':
            if var == 'v':
                surf = axis.pcolormesh(xmesh, ymesh, fld,
                                    norm=matplotlib.colors.LogNorm(),
                                    rcmap='jet_r',rasterized=not self.args.pdf,
                                    shading='gouraud')
            elif var == 'd':
                surf = axis.pcolormesh(xmesh, ymesh, fld,cmap='bwr_r', vmin=0.96, vmax=1.04,
                                    rasterized=not self.args.pdf,
                                    shading='gouraud')
            else:
                surf = axis.pcolormesh(xmesh, ymesh, fld,
                                    rasterized=not self.args.pdf,
                                    shading='gouraud')
            cbar = plt.colorbar(surf, shrink=self.args.shrinkcb)
            cbar.set_label(constants.varlist[var].name)
            plt.axis([self.rcmb, np.amax(xmesh), 0, np.amax(ymesh)])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(misc.file_name(self.args, var).format(self.step) + '.pdf',
                    format='PDF')
        plt.close(fig)

