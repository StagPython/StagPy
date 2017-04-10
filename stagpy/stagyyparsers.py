"""Parsers of StagYY output files"""
from functools import partial
from itertools import product, repeat
from operator import itemgetter
import re
import struct
import numpy as np
import pandas as pd


def time_series(timefile, colnames):
    """Read temporal series from time.dat"""
    if not timefile.is_file():
        return None
    data = np.genfromtxt(timefile, skip_header=1)

    # detect useless lines produced when run is restarted
    rows_to_del = []
    for irow in range(1, len(data)):
        iprev = irow - 1
        while int(data[irow, 0]) <= int(data[iprev, 0]):
            rows_to_del.append(iprev)
            iprev -= 1
    data = np.delete(data, rows_to_del, 0)
    ncols = data.shape[1] - 1
    colnames = colnames[:ncols] + list(range(0, ncols - len(colnames)))

    isteps = map(int, data[:, 0])
    return pd.DataFrame(data[:, 1:], index=isteps, columns=colnames)


def _extract_rsnap_isteps(rproffile):
    """Extract istep and compute list of rows to delete"""
    step_regex = re.compile(r'^\*+step:\s*(\d+) ; time =\s*(\S+)')
    isteps = []  # list of (istep, time, nz)
    rows_to_del = []
    line = ' '
    with rproffile.open() as stream:
        while line[0] != '*':
            line = stream.readline()
        match = step_regex.match(line)
        istep = int(match.group(1))
        time = float(match.group(2))
        nlines = 0
        iline = 0
        for line in stream:
            if line[0] == '*':
                isteps.append((istep, time, nlines))
                match = step_regex.match(line)
                istep = int(match.group(1))
                time = float(match.group(2))
                nlines = 0
                # remove useless lines produced when run is restarted
                nrows_to_del = 0
                while isteps and istep <= isteps[-1][0]:
                    nrows_to_del += isteps.pop()[-1]
                rows_to_del.extend(range(iline - nrows_to_del, iline))
            else:
                nlines += 1
                iline += 1
        isteps.append((istep, time, nlines))
    return isteps, rows_to_del


def rprof(rproffile, colnames):
    """Extract radial profiles data"""
    if not rproffile.is_file():
        return None, None
    data = np.genfromtxt(rproffile, comments='*')

    isteps, rows_to_del = _extract_rsnap_isteps(rproffile)
    data = np.delete(data, rows_to_del, 0)

    ncols = data.shape[1]
    colnames = colnames[:ncols] + list(range(0, ncols - len(colnames)))

    id_arr = [[], []]
    for istep, _, n_z in isteps:
        id_arr[0].extend(repeat(istep, n_z))
        id_arr[1].extend(range(n_z))

    df_rprof = pd.DataFrame(data, index=id_arr, columns=colnames)
    df_times = pd.DataFrame(list(map(itemgetter(1), isteps)),
                            index=map(itemgetter(0), isteps))
    return df_rprof, df_times


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
    if not fieldfile.is_file():
        return None
    header = {}
    with fieldfile.open('rb') as fid:
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
