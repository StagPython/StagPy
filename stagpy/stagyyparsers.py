"""Parsers of StagYY output files"""
from functools import partial
from itertools import product, zip_longest
import re
import struct
import os
import numpy as np


def parse_line(line, convert=None):
    """convert columns of a text line

    line values have to be space separated,
    values are converted to float by default.

    convert argument is a list of functions
    used to convert the first values.
    """
    if convert is None:
        convert = []
    line = line.split()
    for val, func in zip_longest(line, convert[:len(line)], fillvalue=float):
        yield func(val)


def time_series(timefile):
    """Read temporal series from time.dat"""
    # could use np.genfromtxt
    if not os.path.isfile(timefile):
        return None
    with open(timefile, 'r') as infile:
        first = infile.readline()
        data = []
        for line in infile:
            step = list(parse_line(line, convert=[int]))
            # remove useless lines produced when run is restarted
            while data and step[0] <= data[-1][0]:
                data.pop()
            data.append(step)

    # unused at the moment
    colnames = first.split()
    # suppress two columns from the header.
    # Only temporary since this has been corrected in stag
    # WARNING: possibly a problem if some columns are added?
    if len(colnames) == 33:
        colnames = colnames[:28] + colnames[30:]

    return np.array(list(zip_longest(*data, fillvalue=0))).T


def rprof(rproffile):
    """Extract radial profiles data"""
    if not os.path.isfile(rproffile):
        return None
    step_regex = re.compile(r'^\*+step:\s*(\d+) ; time =\s*(\S+)')
    data = []
    data_step = []
    line = ' '
    with open(rproffile) as stream:
        while line[0] != '*':
            line = stream.readline()
        match = step_regex.match(line)
        istep = int(match.group(1))
        time = float(match.group(2))
        # remove useless lines produced when run is restarted
        for line in stream:
            if line[0] == '*':
                while data and istep <= data[-1][0]:
                    data.pop()
                data.append((istep, time, np.array(data_step).T))
                data_step = []
                match = step_regex.match(line)
                istep = int(match.group(1))
                time = float(match.group(2))
            else:
                data_step.append(np.fromstring(line, sep=' '))
        data.append((istep, time, np.array(data_step).T))
    return data


def _readbin(fid, fmt='i', nwords=1, file64=False):
    """Read n words of 4 or 8 bytes with fmt format.

    fmt: 'i' or 'f' (integer or float)
    4 or 8 bytes: depends on header

    Return an array of elements if more than one element.

    Default: read 1 word formatted as an integer.
    """
    if file64:
        nbytes = 8
        fmt = fmt.replace('i', 'q')
        fmt = fmt.replace('f', 'd')
    else:
        nbytes = 4
    elts = np.array(struct.unpack(fmt * nwords, fid.read(nwords * nbytes)))
    if len(elts) == 1:
        elts = elts[0]
    return elts


def fields(fieldfile, only_header=False, only_istep=False):
    """Extract fields data"""
    # something to skip header?
    if not os.path.isfile(fieldfile):
        return None
    header = {}
    with open(fieldfile, 'rb') as fid:
        readbin = partial(_readbin, fid)
        magic = readbin()
        if magic > 8000:  # 64 bits
            magic -= 8000
            readbin()  # need to read 4 more bytes
            readbin = partial(readbin, file64=True)

        # check nb components
        nval = 1
        if magic > 400:
            nval = 4
        elif magic > 300:
            nval = 3

        magic %= 100

        # extra ghost point in horizontal direction
        header['xyp'] = int(magic >= 9 and nval == 4)

        # total number of values in relevant space basis
        # (e1, e2, e3) = (theta, phi, radius) in spherical geometry
        #              = (x, y, z)            in cartesian geometry
        header['nts'] = readbin(nwords=3)

        # number of blocks, 2 for yinyang or cubed sphere
        header['ntb'] = readbin() if magic >= 7 else 1

        # aspect ratio
        header['aspect'] = readbin('f', 2)

        # number of parallel subdomains
        header['ncs'] = readbin(nwords=3)  # (e1, e2, e3) space
        header['ncb'] = readbin() if magic >= 8 else 1  # blocks

        # r - coordinates
        # rgeom[0:self.nrtot+1, 0] are edge radial position
        # rgeom[0:self.nrtot, 1] are cell-center radial position
        if magic >= 2:
            header['rgeom'] = readbin('f', header['nts'][2] * 2 + 1)
        else:
            header['rgeom'] = np.array(range(0, header['nts'][2] * 2 + 1))\
                * 0.5 / header['nts'][2]
        header['rgeom'].resize((header['nts'][2] + 1, 2))

        header['rcmb'] = readbin('f') if magic >= 7 else None

        header['ti_step'] = readbin() if magic >= 3 else 0
        if only_istep:
            return header['ti_step']
        header['ti_ad'] = readbin('f') if magic >= 3 else 0
        header['erupta_total'] = readbin('f') if magic >= 5 else 0
        header['bot_temp'] = readbin('f') if magic >= 6 else 1

        if magic >= 4:
            header['e1_coord'] = readbin('f', header['nts'][0])
            header['e2_coord'] = readbin('f', header['nts'][1])
            header['e3_coord'] = readbin('f', header['nts'][2])
        else:
            # could construct them from other info
            raise ValueError('magic >= 4 expected to get grid geometry')

        if only_header:
            return header

        # READ FIELDS
        # number of points in (e1, e2, e3) directions PER CPU
        npc = header['nts'] // header['ncs']
        # number of blocks per cpu
        nbk = header['ntb'] // header['ncb']
        # number of values per 'read' block
        npi = (npc[0] + header['xyp']) * (npc[1] + header['xyp']) * npc[2] * \
            nbk * nval

        header['scalefac'] = readbin('f') if nval > 1 else 1

        flds = np.zeros((nval,
                         header['nts'][0] + header['xyp'],
                         header['nts'][1] + header['xyp'],
                         header['nts'][2],
                         header['ntb']))

        # loop over parallel subdomains
        for icpu in product(range(header['ncb']),
                            range(header['ncs'][2]),
                            range(header['ncs'][1]),
                            range(header['ncs'][0])):
            # read the data for one CPU
            data_cpu = readbin('f', npi) * header['scalefac']

            # icpu is (icpu block, icpu z, icpu y, icpu x)
            # data from file is transposed to obtained a field
            # array indexed with (x, y, z, block), as in StagYY
            flds[:,
                 icpu[3] * npc[0]:(icpu[3] + 1) * npc[0] + header['xyp'],  # x
                 icpu[2] * npc[1]:(icpu[2] + 1) * npc[1] + header['xyp'],  # y
                 icpu[1] * npc[2]:(icpu[1] + 1) * npc[2],  # z
                 icpu[0] * nbk:(icpu[0] + 1) * nbk  # block
                 ] = np.transpose(data_cpu.reshape(
                     (nbk, npc[2], npc[1] + header['xyp'],
                      npc[0] + header['xyp'], nval)))
    return header, flds
