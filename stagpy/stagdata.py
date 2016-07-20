"""define StagyyData"""

import numpy as np
import re
import struct
from itertools import zip_longest
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
        self.fullname = misc.stag_file(args, self.par_type, timestep)
        self.nval = 4 if self.par_type == 'vp' else 1

        with open(self.fullname, 'rb') as self._fid:
            self._catch_header()
            self._readfile()

    def _readbin(self, fmt='i', nwords=1):
        """Read n words of 4 or 8 bytes with fmt format.

        fmt: 'i' or 'f' (integer or float)
        4 or 8 bytes: depends on header

        Return a tuple of elements if more than one element.

        Default: read 1 word formatted as an integer.
        """
        if self._64bit:
            nbytes = 8
            fmt = fmt.replace('i', 'q')
            fmt = fmt.replace('f', 'd')
        else:
            nbytes = 4
        elts = struct.unpack(fmt * nwords, self._fid.read(nwords * nbytes))
        if len(elts) == 1:
            elts = elts[0]
        return elts

    def _catch_header(self):
        """reads header of binary file"""
        self._64bit = False
        magic = self._readbin()
        if magic > 8000:  # 64 bits
            magic -= 8000
            self._readbin()  # need to read 4 more bytes
            self._64bit = True

        # check nb components
        if magic > 100 and magic // 100 != self.nval:
            raise ValueError('wrong number of components in field')

        magic %= 100

        # extra ghost point in horizontal direction
        self.xyp = int(magic >= 9 and self.nval == 4)

        # total number of values in
        # latitude, longitude and radius directions
        self.nthtot, self.nphtot, self.nrtot = self._readbin(nwords=3)

        # number of blocks, 2 for yinyang
        self.nblocks = self._readbin() if magic >= 7 else 1

        # Aspect ratio
        self.aspect = self._readbin('f', 2)
        self.aspect = np.array(self.aspect)

        # Number of parallel subdomains in the th,ph,r and b directions
        self.nnth, self.nnph, self.nnr = self._readbin(nwords=3)
        self.nnb = self._readbin() if magic >= 8 else 1

        # r - coordinates
        # self.rgeom[0:self.nrtot+1, 0] are edge radial position
        # self.rgeom[0:self.nrtot, 1] are cell-center radial position
        if magic >= 2:
            self.rgeom = np.array(self._readbin('f', self.nrtot * 2 + 1))
        else:
            self.rgeom = np.array(range(0, self.nrtot * 2 + 1))\
                * 0.5 / self.nrtot
        self.rgeom.resize((self.nrtot + 1, 2))

        if magic >= 7:
            self.rcmb = self._readbin('f')  # radius of the cmb
        else:
            self.rcmb = self.args.par_nml['geometry']['r_cmb']
        if magic >= 3:
            self.ti_step = self._readbin()
            self.ti_ad = self._readbin('f')
        else:
            self.ti_step = 0
            self.ti_ad = 0
        self.erupta_total = self._readbin('f') if magic >= 5 else 0
        self.bot_temp = self._readbin('f') if magic >= 6 else 1

        if magic >= 4:
            # theta coordinates
            self.th_coord = np.array(self._readbin('f', self.nthtot))
            # force to pi/2 if 2D
            self.th_coord = np.array(np.pi / 2)
            # phi coordinates
            ph_coord = np.array(self._readbin('f', self.nphtot))
            self._ph_coord = ph_coord
            # to have continuous field
            self.ph_coord = np.append(ph_coord, ph_coord[1] - ph_coord[0])
            # radius coordinates
            self.r_coord = np.array(self._readbin('f', self.nrtot))
        else:
            # could construct them from other info
            raise ValueError('magic >= 4 expected to get grid geometry')

        # create meshgrids
        self.th_mesh, self.ph_mesh, self.r_mesh = np.meshgrid(
            self.th_coord, self.ph_coord, self.r_coord + self.rcmb,
            indexing='ij')

        # compute cartesian coordinates
        # z along rotation axis at theta=0
        # x at th=90, phi=0
        # y at th=90, phi=90
        self.x_mesh = self.r_mesh * np.cos(self.ph_mesh) * np.sin(self.th_mesh)
        self.y_mesh = self.r_mesh * np.sin(self.ph_mesh) * np.sin(self.th_mesh)
        self.z_mesh = self.r_mesh * np.cos(self.th_mesh)

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

        # flds should be construct with the "normal" indexing order th, ph, r
        # there shouldn't be a need to transpose in plot_scalar
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
        vphi = self.fields['v'][:, :, 0]
        # interpolate to the same phi
        vph2 = -0.5 * (vphi + np.roll(vphi, 1, 1))
        v_r = self.fields['w'][:, :, 0]
        n_r, nph = np.shape(v_r)
        stream = np.zeros(np.shape(vphi))
        # integrate first on phi
        stream[0, 1:nph - 1] = self.rcmb * \
            integrate.cumtrapz(v_r[0, 0:nph - 1], self._ph_coord)
        stream[0, 0] = 0
        # use r coordinates where vphi is defined
        rcoord = self.rcmb + np.array(
            self.rgeom[0:np.shape(self.rgeom)[0] - 1:2])
        for iph in range(0, np.shape(vph2)[1] - 1):
            stream[1:n_r, iph] = stream[0, iph] + \
                integrate.cumtrapz(vph2[:, iph], rcoord)  # integrate on r
        stream = stream - np.mean(stream[n_r / 2, :])
        # remove some typical value.
        # Would be better to compute the global average
        # taking into account variable grid spacing
        return stream


class RprofData:

    """extract radial profiles data"""

    def __init__(self, args):
        """create RprofData object"""
        step_regex = re.compile(r'^\*+step:\s*(\d+) ; time =\s*(\S+)')
        self._readproffile(args, step_regex)

    def _readproffile(self, args, step_regex):
        """extract info from rprof.dat"""
        proffile = misc.stag_file(args, 'rprof.dat')
        timesteps = []
        data0 = []
        lnum = -1
        with open(proffile) as stream:
            for line in stream:
                if line != '\n':
                    lnum += 1
                    if line[0] == '*':
                        match = step_regex.match(line)
                        timesteps.append([lnum, int(match.group(1)),
                                          float(match.group(2))])
                    else:
                        data0.append(np.array(line.split()))
        tsteps = np.array(timesteps)
        nsteps = tsteps.shape[0]
        data = np.array(data0)
        # all the processing of timesteps
        # should be in commands.*_cmd
        # instead of main.py
        # since it could be different between
        # the different modules
        istart, ilast, istep = args.timestep
        if ilast == -1:
            ilast = nsteps - 1
        if istart == -1:
            istart = nsteps - 1
        args.timestep = istart, ilast, istep

        # number of points for each profile
        nzp = []
        for iti in range(0, nsteps - 1):
            nzp = np.append(nzp, tsteps[iti + 1, 0] - tsteps[iti, 0] - 1)
        nzp = np.append(nzp, lnum - tsteps[nsteps - 1, 0])

        nzs = [[0, 0, 0]]
        nzc = 0
        for iti in range(1, nsteps):
            if nzp[iti] != nzp[iti - 1]:
                nzs.append([iti, iti - nzc, int(nzp[iti - 1])])
                nzc = iti
        if nzp[nsteps - 1] != nzs[-1][1]:
            nzs.append([nsteps, nsteps - nzc, int(nzp[nsteps - 1])])
        nzi = np.array(nzs)
        self.data = data  # contains the actual profile data
        # line number, timestep number, time for each profile
        self.tsteps = tsteps
        self.nzi = nzi
        # stores the profile numbers where number points changes,
        # number of profiles with that number of points, and number of points


class TimeData:

    """extract temporal series"""

    def __init__(self, args):
        """read temporal series from time.dat"""
        timefile = misc.stag_file(args, 'time.dat')
        with open(timefile, 'r') as infile:
            first = infile.readline()
            line = infile.readline()
            data = [list(misc.parse_line(line))]
            for line in infile:
                step = list(misc.parse_line(line, convert=[int]))
                # remove useless lines when run is restarted
                while step[0] <= data[-1][0]:
                    data.pop()
                data.append(step)

        self.colnames = first.split()
        # suppress two columns from the header.
        # Only temporary since this has been corrected in stag
        # WARNING: possibly a problem if some columns are added?
        if len(self.colnames) == 33:
            self.colnames = self.colnames[:28] + self.colnames[30:]

        self.data = np.array(list(zip_longest(*data, fillvalue=0))).T
