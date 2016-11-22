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
        self.nval = 4 if (self.par_type == 'vp' or self.par_type == 'sx') else 1

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

        # total number of values in relevant space basis
        # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
        #              = (x, y, z)            in cartesian geometry
        self.nts = np.array(self._readbin(nwords=3))

        # number of blocks, 2 for yinyang or cubed sphere
        self.ntb = self._readbin() if magic >= 7 else 1

        # Aspect ratio
        self.aspect = np.array(self._readbin('f', 2))

        # Number of parallel subdomains
        self.ncs = np.array(self._readbin(nwords=3))  # (e1, e2, e3) space
        self.ncb = self._readbin() if magic >= 8 else 1  # blocks

        # r - coordinates
        # self.rgeom[0:self.nrtot+1, 0] are edge radial position
        # self.rgeom[0:self.nrtot, 1] are cell-center radial position
        if magic >= 2:
            self.rgeom = np.array(self._readbin('f', self.nts[2] * 2 + 1))
        else:
            self.rgeom = np.array(range(0, self.nts[2] * 2 + 1))\
                * 0.5 / self.nts[2]
        self.rgeom.resize((self.nts[2] + 1, 2))

        if magic >= 7:
            self.rcmb = self._readbin('f')  # radius of the cmb
            if self.rcmb == -1:
                self.geometry = 'cartesian'  # need true geometry descriptor
            else:
                self.geometry = 'curvilinear'
        else:
            # can't tell anything about geometry...
            # need to infer it from par
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
            e1_coord = np.array(self._readbin('f', self.nts[0]))
            e2_coord = np.array(self._readbin('f', self.nts[1]))
            e3_coord = np.array(self._readbin('f', self.nts[2]))
        else:
            # could construct them from other info
            raise ValueError('magic >= 4 expected to get grid geometry')

        if self.rcmb == -1:
            # cartesian
            self.nxtot = self.nts[0]
            self.nytot = self.nts[1]
            self.nztot = self.nts[2]
            self.x_coord = e1_coord
            self.y_coord = e2_coord
            self.z_coord = e3_coord

            self.x_mesh, self.y_mesh, self.z_mesh = np.meshgrid(
                self.x_coord, self.y_coord, self.z_coord, indexing='ij')
        else:
            # spherical
            self.nthtot = self.nts[0]
            self.nphtot = self.nts[1]
            self.nrtot = self.nts[2]
            self.th_coord = e1_coord
            self.ph_coord = e2_coord
            self.r_coord = e3_coord
            if self.nts[0] == 1:
                # one point in theta: spherical annulus
                self.th_coord = np.array(np.pi / 2)
                self._ph_coord = e2_coord
                # to have continuous field
                self.ph_coord = np.append(e2_coord, e2_coord[1] - e2_coord[0])

            th_mesh, ph_mesh, r_mesh = np.meshgrid(
                self.th_coord, self.ph_coord, self.r_coord + self.rcmb,
                indexing='ij')

            # compute cartesian coordinates
            # z along rotation axis at theta=0
            # x at th=90, phi=0
            # y at th=90, phi=90
            self.x_mesh = r_mesh * np.cos(ph_mesh) * np.sin(th_mesh)
            self.y_mesh = r_mesh * np.sin(ph_mesh) * np.sin(th_mesh)
            self.z_mesh = r_mesh * np.cos(th_mesh)

    def _readfile(self):
        """read scalar/vector fields"""
        # compute number of points in (e1, e2, e3) directions PER CPU
        ne1, ne2, ne3 = self.nts // self.ncs
        # compute number of blocks per cpu
        nbk = self.ntb // self.ncb
        # the number of values per 'read' block
        npi = (ne1 + self.xyp) * (ne2 + self.xyp) * ne3 * nbk * self.nval

        if self.nval > 1:
            self.scalefac = self._readbin('f')
        else:
            self.scalefac = 1

        # flds should be constructed with the "normal" indexing
        # order (e1, e2, e3).
        # There shouldn't be a need to transpose in plot_scalar.
        dim_fields = (self.ntb, self.nts[2],
                      self.nts[1] + self.xyp, self.nts[0] + self.xyp)

        flds = []
        for _ in range(self.nval):
            flds.append(np.zeros(dim_fields))

        # loop over parallel subdomains
        for icbk in np.arange(self.ncb):
            for ice3 in np.arange(self.ncs[2]):
                for ice2 in np.arange(self.ncs[1]):
                    for ice1 in np.arange(self.ncs[0]):
                        # read the data for this CPU
                        file_content = self._readbin('f', npi)
                        data_cpu = np.array(file_content) * self.scalefac

                        # Create a 3D matrix from these data
                        data_cpu_3d = data_cpu.reshape(
                            (nbk, ne3, ne2 + self.xyp,
                             ne1 + self.xyp, self.nval))

                        # Add local 3D matrix to global matrix
                        se1 = ice1 * ne1
                        ee1 = ice1 * ne1 + ne1 + self.xyp
                        se2 = ice2 * ne2
                        ee2 = ice2 * ne2 + ne2 + self.xyp
                        se3 = ice3 * ne3
                        ee3 = ice3 * ne3 + ne3
                        sbk = icbk * nbk
                        ebk = icbk * nbk + nbk

                        for idx, fld in enumerate(flds):
                            fld[sbk:ebk, se3:ee3, se2:ee2, se1:ee1] = \
                                data_cpu_3d[:, :, :, :, idx]

        self.fields = {}
        if self.par_type == 'vp':
            fld_names = ['u', 'v', 'w', 'p']
        elif self.par_type == 'sx':
            fld_names = ['sx', 'sy', 'sz', 'x']
        else:
            fld_names = [self.var]
        for fld_name, fld in zip(fld_names, flds):
            if self.ntb == 1:
                self.fields[fld_name] = fld[0, :, :, :]
            else:
                self.fields[fld_name] = fld[:, :, :, :]

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
        rcoord = self.rcmb + self.rgeom[0:self.nrtot, 1]
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
