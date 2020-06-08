"""Parsers of StagYY output files.

Note:
    These functions are low level utilities. You should not use these unless
    you know what you are doing. To access StagYY output data, use an instance
    of :class:`~stagpy.stagyydata.StagyyData`.
"""
from functools import partial
from itertools import product, repeat
from operator import itemgetter
from xml.etree import ElementTree as xmlET
import re

import numpy as np
import pandas as pd
import h5py

from .error import ParsingError
from .phyvars import FIELD_FILES_H5


def _tidy_names(names, nnames, extra_names=None):
    """Truncate or extend names so that its len is nnames.

    The list is modified, this function returns nothing.

    Args:
        names (list): list of names.
        nnames (int): desired number of names.
        extra_names (list of str): list of names to be used to extend the list
            if needed. If this list isn't provided, a range is used instead.
    """
    if len(names) < nnames and extra_names is not None:
        names.extend(extra_names)
    names.extend(range(nnames - len(names)))
    del names[nnames:]


def time_series(timefile, colnames):
    """Read temporal series text file.

    If :data:`colnames` is too long, it will be truncated. If it is too short,
    additional numeric column names from 0 to N-1 will be attributed to the N
    extra columns present in :data:`timefile`.

    Args:
        timefile (:class:`pathlib.Path`): path of the time.dat file.
        colnames (list of names): names of the variables expected in
            :data:`timefile` (may be modified).

    Returns:
        :class:`pandas.DataFrame`:
            Time series, with the variables in columns and the time steps in
            rows.
    """
    if not timefile.is_file():
        return None
    data = pd.read_csv(timefile, delim_whitespace=True, dtype=str,
                       header=None, skiprows=1, index_col=0,
                       engine='c', memory_map=True,
                       error_bad_lines=False, warn_bad_lines=False)
    data = data.apply(pd.to_numeric, raw=True, errors='coerce')

    # detect useless lines produced when run is restarted
    rows_to_del = []
    irow = len(data) - 1
    while irow > 0:
        iprev = irow - 1
        while iprev >= 0 and data.index[irow] <= data.index[iprev]:
            rows_to_del.append(iprev)
            iprev -= 1
        irow = iprev
    if rows_to_del:
        rows_to_keep = set(range(len(data))) - set(rows_to_del)
        data = data.take(list(rows_to_keep))

    ncols = data.shape[1]
    _tidy_names(colnames, ncols)
    data.columns = colnames

    return data


def time_series_h5(timefile, colnames):
    """Read temporal series HDF5 file.

    If :data:`colnames` is too long, it will be truncated. If it is too short,
    additional column names will be deduced from the content of the file.

    Args:
        timefile (:class:`pathlib.Path`): path of the TimeSeries.h5 file.
        colnames (list of names): names of the variables expected in
            :data:`timefile` (may be modified).

    Returns:
        :class:`pandas.DataFrame`:
            Time series, with the variables in columns and the time steps in
            rows.
    """
    if not timefile.is_file():
        return None
    with h5py.File(timefile, 'r') as h5f:
        dset = h5f['tseries']
        _, ncols = dset.shape
        ncols -= 1  # first is istep
        h5names = map(bytes.decode, h5f['names'][len(colnames) + 1:])
        _tidy_names(colnames, ncols, h5names)
        data = dset[()]
    pdf = pd.DataFrame(data[:, 1:],
                       index=np.int_(data[:, 0]), columns=colnames)
    # remove duplicated lines in case of restart
    return pdf.loc[~pdf.index.duplicated(keep='last')]


def _extract_rsnap_isteps(rproffile):
    """Extract istep and compute list of rows to delete."""
    step_regex = re.compile(r'^\*+step:\s*(\d+) ; time =\s*(\S+)')
    isteps = []  # list of (istep, time, nz)
    rows_to_del = set()
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
                rows_to_del = rows_to_del.union(
                    range(iline - nrows_to_del, iline))
            else:
                nlines += 1
                iline += 1
        isteps.append((istep, time, nlines))
    return isteps, rows_to_del


def rprof(rproffile, colnames):
    """Extract radial profiles data.

    If :data:`colnames` is too long, it will be truncated. If it is too short,
    additional numeric column names from 0 to N-1 will be attributed to the N
    extra columns present in :data:`timefile`.

    Args:
        rproffile (:class:`pathlib.Path`): path of the rprof.dat file.
        colnames (list of names): names of the variables expected in
            :data:`rproffile` (may be modified).

    Returns:
        tuple of :class:`pandas.DataFrame`: (profs, times)
            :data:`profs` are the radial profiles, with the variables in
            columns and rows double-indexed with the time step and the radial
            index of numerical cells.

            :data:`times` is the dimensionless time indexed by time steps.
    """
    if not rproffile.is_file():
        return None, None
    data = pd.read_csv(rproffile, delim_whitespace=True, dtype=str,
                       header=None, comment='*', skiprows=1,
                       engine='c', memory_map=True,
                       error_bad_lines=False, warn_bad_lines=False)
    data = data.apply(pd.to_numeric, raw=True, errors='coerce')

    isteps, rows_to_del = _extract_rsnap_isteps(rproffile)
    if rows_to_del:
        rows_to_keep = set(range(len(data))) - rows_to_del
        data = data.take(list(rows_to_keep))

    id_arr = [[], []]
    for istep, _, n_z in isteps:
        id_arr[0].extend(repeat(istep, n_z))
        id_arr[1].extend(range(n_z))

    data.index = id_arr

    ncols = data.shape[1]
    _tidy_names(colnames, ncols)
    data.columns = colnames

    df_times = pd.DataFrame(list(map(itemgetter(1), isteps)),
                            index=map(itemgetter(0), isteps))
    return data, df_times


def rprof_h5(rproffile, colnames):
    """Extract radial profiles data.

    If :data:`colnames` is too long, it will be truncated. If it is too short,
    additional column names will be deduced from the content of the file.

    Args:
        rproffile (:class:`pathlib.Path`): path of the rprof.dat file.
        colnames (list of names): names of the variables expected in
            :data:`rproffile`.

    Returns:
        tuple of :class:`pandas.DataFrame`: (profs, times)
            :data:`profs` are the radial profiles, with the variables in
            columns and rows double-indexed with the time step and the radial
            index of numerical cells.

            :data:`times` is the dimensionless time indexed by time steps.
    """
    if not rproffile.is_file():
        return None, None
    isteps = []
    with h5py.File(rproffile, 'r') as h5f:
        dnames = sorted(dname for dname in h5f.keys()
                        if dname.startswith('rprof_'))
        ncols = h5f['names'].shape[0]
        h5names = map(bytes.decode, h5f['names'][len(colnames):])
        _tidy_names(colnames, ncols, h5names)
        data = np.zeros((0, ncols))
        for dname in dnames:
            dset = h5f[dname]
            data = np.concatenate((data, dset[()]))
            isteps.append((dset.attrs['istep'], dset.attrs['time'],
                           dset.shape[0]))

    id_arr = [[], []]
    for istep, _, n_z in isteps:
        id_arr[0].extend(repeat(istep, n_z))
        id_arr[1].extend(range(n_z))

    df_data = pd.DataFrame(data, index=id_arr, columns=colnames)
    df_times = pd.DataFrame(list(map(itemgetter(1), isteps)),
                            index=map(itemgetter(0), isteps))
    return df_data, df_times


def _clean_names_refstate(names):
    """Uniformization of refstate profile names."""
    to_clean = {
        'Tref': 'T',
        'rhoref': 'rho',
        'tcond': 'Tcond',
    }
    return [to_clean.get(n, n) for n in names]


def refstate(reffile, ncols=8):
    """Extract reference state profiles.

    Args:
        reffile (:class:`pathlib.Path`): path of the refstate file.
        ncols (int): number of columns.

    Returns:
        tuple of list: (syst, adia)
            :data:`syst` is a list of list of :class:`pandas.DataFrame`
            containing the reference state profiles for each system and
            each phase in these systems.

            :data:`adia` is a list of :class:`pandas.DataFrame` containing
            the adiabatic reference state profiles for each system, the last
            item being the combined adiabat.
    """
    if not reffile.is_file():
        return None, None
    data = pd.read_csv(reffile, delim_whitespace=True, dtype=str,
                       header=None, names=range(ncols),
                       engine='c', memory_map=True,
                       error_bad_lines=False, warn_bad_lines=False)
    data = data.apply(pd.to_numeric, raw=True, errors='coerce')
    # drop lines corresponding to metadata
    data.dropna(subset=[0], inplace=True)
    isystem = -1
    systems = [[]]
    adiabats = []
    with reffile.open() as rsf:
        for line in rsf:
            line = line.lstrip()
            if line.startswith('SYSTEM'):
                isystem += 1
                if isystem > 0:
                    systems.append([])
            elif line.startswith('z'):
                systems[isystem].append(_clean_names_refstate(line.split()))
            elif line.startswith('ADIABAT') or line.startswith('COMBINED'):
                line = line.partition(':')[-1]
                adiabats.append(_clean_names_refstate(line.split()))
    nprofs = sum(map(len, systems)) + len(adiabats)
    nzprof = len(data) // nprofs
    iprof = 0
    syst = []
    adia = []
    for isys, layers in enumerate(systems):
        syst.append([])
        for layer in layers:
            ibgn = iprof * nzprof
            iend = ibgn + nzprof
            syst[isys].append(
                pd.DataFrame(data.iloc[ibgn:iend, :len(layer)].values,
                             columns=layer))
            iprof += 1
        if len(layers) > 1:
            ibgn = iprof * nzprof
            iend = ibgn + nzprof
            cols = adiabats.pop(0)
            adia.append(
                pd.DataFrame(data.iloc[ibgn:iend, :len(cols)].values,
                             columns=cols))
            iprof += 1
        else:
            adia.append(syst[isys][0])
    ibgn = iprof * nzprof
    iend = ibgn + nzprof
    cols = adiabats.pop(0)
    adia.append(pd.DataFrame(data.iloc[ibgn:iend, :len(cols)].values,
                             columns=cols))
    return syst, adia


def _readbin(fid, fmt='i', nwords=1, file64=False, unpack=True):
    """Read n words of 4 or 8 bytes with fmt format.

    fmt: 'i' or 'f' or 'b' (integer or float or bytes)
    4 or 8 bytes: depends on header

    Return an array of elements if more than one element.

    Default: read 1 word formatted as an integer.
    """
    if fmt in 'if':
        fmt += '8' if file64 else '4'
    elts = np.fromfile(fid, fmt, nwords)
    if unpack and len(elts) == 1:
        elts = elts[0]
    return elts


def fields(fieldfile, only_header=False, only_istep=False):
    """Extract fields data.

    Args:
        fieldfile (:class:`pathlib.Path`): path of the binary field file.
        only_header (bool): when True (and :data:`only_istep` is False), only
            :data:`header` is returned.
        only_istep (bool): when True, only :data:`istep` is returned.

    Returns:
        depends on flags.: :obj:`int`: istep
            If :data:`only_istep` is True, this function returns the time step
            at which the binary file was written.
        :obj:`dict`: header
            Else, if :data:`only_header` is True, this function returns a dict
            containing the header informations of the binary file.
        :class:`numpy.array`: fields
            Else, this function returns the tuple :data:`(header, fields)`.
            :data:`fields` is an array of scalar fields indexed by variable,
            x-direction, y-direction, z-direction, block.
    """
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
        sfield = False
        if magic > 400:
            nval = 4
        elif magic > 300:
            nval = 3
        elif magic > 100:
            sfield = True

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
        header['rgeom'] = np.resize(header['rgeom'], (header['nts'][2] + 1, 2))

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
            raise ParsingError(fieldfile,
                               'magic >= 4 expected to get grid geometry')

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
        if sfield:
            # for surface fields, variables are written along z direction
            flds = np.swapaxes(flds, 0, 3)
    return header, flds


def tracers(tracersfile):
    """Extract tracers data.

    Args:
        tracersfile (:class:`pathlib.Path`): path of the binary tracers file.

    Returns:
        dict of list of numpy.array:
            Tracers data organized by attribute and block.
    """
    if not tracersfile.is_file():
        return None
    tra = {}
    with tracersfile.open('rb') as fid:
        readbin = partial(_readbin, fid)
        magic = readbin()
        if magic > 8000:  # 64 bits
            magic -= 8000
            readbin()
            readbin = partial(readbin, file64=True)
        if magic < 100:
            raise ParsingError(tracersfile,
                               'magic > 100 expected to get tracervar info')
        nblk = magic % 100
        readbin('f', 2)  # aspect ratio
        readbin()  # istep
        readbin('f')  # time
        ninfo = readbin()
        ntra = readbin(nwords=nblk, unpack=False)
        readbin('f')  # tracer ideal mass
        curv = readbin()
        if curv:
            readbin('f')  # r_cmb
        infos = []  # list of info names
        for _ in range(ninfo):
            infos.append(b''.join(readbin('b', 16)).strip().decode())
            tra[infos[-1]] = []
        if magic > 200:
            ntrace_elt = readbin()
            if ntrace_elt > 0:
                readbin('f', ntrace_elt)  # outgassed
        for ntrab in ntra:  # blocks
            data = readbin('f', ntrab * ninfo)
            for idx, info in enumerate(infos):
                tra[info].append(data[idx::ninfo])
    return tra


def _read_group_h5(filename, groupname):
    """Return group content.

    Args:
        filename (:class:`pathlib.Path`): path of hdf5 file.
        groupname (str): name of group to read.
    Returns:
        :class:`numpy.array`: content of group.
    """
    with h5py.File(filename, 'r') as h5f:
        data = h5f[groupname][()]
    return data  # need to be reshaped


def _make_3d(field, twod):
    """Add a dimension to field if necessary.

    Args:
        field (numpy.array): the field that need to be 3d.
        twod (str): 'XZ', 'YZ' or None depending on what is relevant.
    Returns:
        numpy.array: reshaped field.
    """
    shp = list(field.shape)
    if twod and 'X' in twod:
        shp.insert(1, 1)
    elif twod:
        shp.insert(0, 1)
    return field.reshape(shp)


def _ncores(meshes, twod):
    """Compute number of nodes in each direction."""
    nnpb = len(meshes)  # number of nodes per block
    nns = [1, 1, 1]  # number of nodes in x, y, z directions
    if twod is None or 'X' in twod:
        while (nnpb > 1 and
               meshes[nns[0]]['X'][0, 0, 0] ==
               meshes[nns[0] - 1]['X'][-1, 0, 0]):
            nns[0] += 1
            nnpb -= 1
    cpu = lambda icy: icy * nns[0]
    if twod is None or 'Y' in twod:
        while (nnpb > 1 and
               meshes[cpu(nns[1])]['Y'][0, 0, 0] ==
               meshes[cpu(nns[1] - 1)]['Y'][0, -1, 0]):
            nns[1] += 1
            nnpb -= nns[0]
    cpu = lambda icz: icz * nns[0] * nns[1]
    while (nnpb > 1 and
           meshes[cpu(nns[2])]['Z'][0, 0, 0] ==
           meshes[cpu(nns[2] - 1)]['Z'][0, 0, -1]):
        nns[2] += 1
        nnpb -= nns[0] * nns[1]
    return np.array(nns)


def _conglomerate_meshes(meshin, header):
    """Conglomerate meshes from several cores into one."""
    meshout = {}
    npc = header['nts'] // header['ncs']
    shp = [val + 1 if val != 1 else 1 for val in header['nts']]
    x_p = int(shp[0] != 1)
    y_p = int(shp[1] != 1)
    for coord in meshin[0]:
        meshout[coord] = np.zeros(shp)
    for icore in range(np.prod(header['ncs'])):
        ifs = [icore // np.prod(header['ncs'][:i]) % header['ncs'][i] * npc[i]
               for i in range(3)]
        for coord, mesh in meshin[icore].items():
            meshout[coord][ifs[0]:ifs[0] + npc[0] + x_p,
                           ifs[1]:ifs[1] + npc[1] + y_p,
                           ifs[2]:ifs[2] + npc[2] + 1] = mesh
    return meshout


def _read_coord_h5(files, shapes, header, twod):
    """Read all coord hdf5 files of a snapshot.

    Args:
        files (list of pathlib.Path): list of NodeCoordinates files of
            a snapshot.
        shapes (list of (int,int)): shape of mesh grids.
        header (dict): geometry info.
        twod (str): 'XZ', 'YZ' or None depending on what is relevant.
    """
    meshes = []
    for h5file, shape in zip(files, shapes):
        meshes.append({})
        with h5py.File(h5file, 'r') as h5f:
            for coord, mesh in h5f.items():
                # for some reason, the array is transposed!
                meshes[-1][coord] = mesh[()].reshape(shape).T
                meshes[-1][coord] = _make_3d(meshes[-1][coord], twod)

    header['ncs'] = _ncores(meshes, twod)
    header['nts'] = list((meshes[0]['X'].shape[i] - 1) * header['ncs'][i]
                         for i in range(3))
    header['nts'] = np.array([max(1, val) for val in header['nts']])
    # meshes could also be defined in legacy parser, so that these can be used
    # in geometry setup
    meshes = _conglomerate_meshes(meshes, header)
    if np.any(meshes['Z'][:, :, 0] != 0):
        # spherical
        header['x_mesh'] = np.copy(meshes['Y'])  # annulus geometry...
        header['y_mesh'] = np.copy(meshes['Z'])
        header['z_mesh'] = np.copy(meshes['X'])
        header['r_mesh'] = np.sqrt(header['x_mesh']**2 + header['y_mesh']**2 +
                                   header['z_mesh']**2)
        header['t_mesh'] = np.arccos(header['z_mesh'] / header['r_mesh'])
        header['p_mesh'] = np.roll(
            np.arctan2(header['y_mesh'], -header['x_mesh']) + np.pi, -1, 1)
        header['e1_coord'] = header['t_mesh'][:, 0, 0]
        header['e2_coord'] = header['p_mesh'][0, :, 0]
        header['e3_coord'] = header['r_mesh'][0, 0, :]
    else:
        header['e1_coord'] = meshes['X'][:, 0, 0]
        header['e2_coord'] = meshes['Y'][0, :, 0]
        header['e3_coord'] = meshes['Z'][0, 0, :]
    header['aspect'] = (header['e1_coord'][-1] - header['e2_coord'][0],
                        header['e1_coord'][-1] - header['e2_coord'][0])
    header['rcmb'] = header['e3_coord'][0]
    if header['rcmb'] == 0:
        header['rcmb'] = -1
    else:
        header['e3_coord'] = header['e3_coord'] - header['rcmb']
    if twod is None or 'X' in twod:
        header['e1_coord'] = header['e1_coord'][:-1]
    if twod is None or 'Y' in twod:
        header['e2_coord'] = header['e2_coord'][:-1]
    header['e3_coord'] = header['e3_coord'][:-1]


def _get_dim(data_item):
    """Extract shape of data item."""
    return tuple(map(int, data_item.get('Dimensions').split()))


def _get_field(xdmf_file, data_item):
    """Extract field from data item."""
    shp = _get_dim(data_item)
    h5file, group = data_item.text.strip().split(':/', 1)
    icore = int(group.split('_')[-2]) - 1
    fld = _read_group_h5(xdmf_file.parent / h5file, group).reshape(shp)
    return icore, fld


def _maybe_get(elt, item, info, conversion=None):
    """Extract and convert info if item is present."""
    maybe_item = elt.find(item)
    if maybe_item is not None:
        maybe_item = maybe_item.get(info)
        if conversion is not None:
            maybe_item = conversion(maybe_item)
    return maybe_item


def read_geom_h5(xdmf_file, snapshot):
    """Extract geometry information from hdf5 files.

    Args:
        xdmf_file (:class:`pathlib.Path`): path of the xdmf file.
        snapshot (int): snapshot number.
    Returns:
        (dict, root): geometry information and root of xdmf document.
    """
    header = {}
    xdmf_root = xmlET.parse(str(xdmf_file)).getroot()
    if snapshot is None:
        return None, xdmf_root

    # Domain, Temporal Collection, Snapshot
    # should check that this is indeed the required snapshot
    elt_snap = xdmf_root[0][0][snapshot]
    header['ti_ad'] = float(elt_snap.find('Time').get('Value'))
    header['mo_lambda'] = _maybe_get(elt_snap, 'mo_lambda', 'Value', float)
    header['mo_thick_sol'] = _maybe_get(elt_snap, 'mo_thick_sol', 'Value',
                                        float)
    header['ntb'] = 1
    coord_h5 = []  # all the coordinate files
    coord_shape = []  # shape of meshes
    twod = None
    for elt_subdomain in elt_snap.findall('Grid'):
        if elt_subdomain.get('Name').startswith('meshYang'):
            header['ntb'] = 2
            break  # iterate only through meshYin
        elt_geom = elt_subdomain.find('Geometry')
        if elt_geom.get('Type') == 'X_Y' and twod is None:
            twod = ''
            for data_item in elt_geom.findall('DataItem'):
                coord = data_item.text.strip()[-1]
                if coord in 'XYZ':
                    twod += coord
        data_item = elt_geom.find('DataItem')
        coord_shape.append(_get_dim(data_item))
        coord_h5.append(
            xdmf_file.parent / data_item.text.strip().split(':/', 1)[0])
    _read_coord_h5(coord_h5, coord_shape, header, twod)
    return header, xdmf_root


def _to_spherical(flds, header):
    """Convert vector field to spherical."""
    cth = np.cos(header['t_mesh'][:, :, :-1])
    sth = np.sin(header['t_mesh'][:, :, :-1])
    cph = np.cos(header['p_mesh'][:, :, :-1])
    sph = np.sin(header['p_mesh'][:, :, :-1])
    fout = np.copy(flds)
    fout[0] = cth * cph * flds[0] + cth * sph * flds[1] - sth * flds[2]
    fout[1] = sph * flds[0] - cph * flds[1]  # need to take the opposite here
    fout[2] = sth * cph * flds[0] + sth * sph * flds[1] + cth * flds[2]
    return fout


def _flds_shape(fieldname, header):
    """Compute shape of flds variable."""
    shp = list(header['nts'])
    shp.append(header['ntb'])
    if len(FIELD_FILES_H5[fieldname]) == 3:
        shp.insert(0, 3)
        # extra points
        header['xp'] = int(header['nts'][0] != 1)
        shp[1] += header['xp']
        header['yp'] = int(header['nts'][1] != 1)
        shp[2] += header['yp']
        header['zp'] = 1
        header['xyp'] = 1
    else:
        shp.insert(0, 1)
        header['xp'] = 0
        header['yp'] = 0
        header['zp'] = 0
        header['xyp'] = 0
    return shp


def _post_read_flds(flds, header):
    """Process flds to handle sphericity."""
    if flds.shape[0] >= 3 and header['rcmb'] > 0:
        # spherical vector
        header['p_mesh'] = np.roll(
            np.arctan2(header['y_mesh'], header['x_mesh']), -1, 1)
        for ibk in range(header['ntb']):
            flds[..., ibk] = _to_spherical(flds[..., ibk], header)
        header['p_mesh'] = np.roll(
            np.arctan2(header['y_mesh'], -header['x_mesh']) + np.pi, -1, 1)
    return flds


def read_field_h5(xdmf_file, fieldname, snapshot, header=None):
    """Extract field data from hdf5 files.

    Args:
        xdmf_file (:class:`pathlib.Path`): path of the xdmf file.
        fieldname (str): name of field to extract.
        snapshot (int): snapshot number.
        header (dict): geometry information.
    Returns:
        (dict, numpy.array): geometry information and field data. None
            is returned if data is unavailable.
    """
    if header is None:
        header, xdmf_root = read_geom_h5(xdmf_file, snapshot)
    else:
        xdmf_root = xmlET.parse(str(xdmf_file)).getroot()

    npc = header['nts'] // header['ncs']  # number of grid point per node
    flds = np.zeros(_flds_shape(fieldname, header))
    data_found = False

    for elt_subdomain in xdmf_root[0][0][snapshot].findall('Grid'):
        ibk = int(elt_subdomain.get('Name').startswith('meshYang'))
        for data_attr in elt_subdomain.findall('Attribute'):
            if data_attr.get('Name') != fieldname:
                continue
            icore, fld = _get_field(xdmf_file, data_attr.find('DataItem'))
            # for some reason, the field is transposed
            fld = fld.T
            shp = fld.shape
            if shp[-1] == 1 and header['nts'][0] == 1:  # YZ
                fld = fld.reshape((shp[0], 1, shp[1], shp[2]))
                if header['rcmb'] < 0:
                    fld = fld[(2, 0, 1), ...]
            elif shp[-1] == 1:  # XZ
                fld = fld.reshape((shp[0], shp[1], 1, shp[2]))
                if header['rcmb'] < 0:
                    fld = fld[(1, 2, 0), ...]
            elif header['nts'][1] == 1:  # cart XZ
                fld = fld.reshape((1, shp[0], 1, shp[1]))
            ifs = [icore // np.prod(header['ncs'][:i]) % header['ncs'][i] *
                   npc[i] for i in range(3)]
            if header['zp']:  # remove top row
                fld = fld[:, :, :, :-1]
            flds[:,
                 ifs[0]:ifs[0] + npc[0] + header['xp'],
                 ifs[1]:ifs[1] + npc[1] + header['yp'],
                 ifs[2]:ifs[2] + npc[2],
                 ibk] = fld
            data_found = True

    flds = _post_read_flds(flds, header)

    return (header, flds) if data_found else None


def read_tracers_h5(xdmf_file, infoname, snapshot, position):
    """Extract tracers data from hdf5 files.

    Args:
        xdmf_file (:class:`pathlib.Path`): path of the xdmf file.
        infoname (str): name of information to extract.
        snapshot (int): snapshot number.
        position (bool): whether to extract position of tracers.
    Returns:
        dict of list of numpy.array:
            Tracers data organized by attribute and block.
    """
    xdmf_root = xmlET.parse(str(xdmf_file)).getroot()
    tra = {}
    tra[infoname] = [{}, {}]  # two blocks, ordered by cores
    if position:
        for axis in 'xyz':
            tra[axis] = [{}, {}]
    for elt_subdomain in xdmf_root[0][0][snapshot].findall('Grid'):
        ibk = int(elt_subdomain.get('Name').startswith('meshYang'))
        if position:
            for data_attr in elt_subdomain.findall('Geometry'):
                for data_item, axis in zip(data_attr.findall('DataItem'),
                                           'xyz'):
                    icore, data = _get_field(xdmf_file, data_item)
                    tra[axis][ibk][icore] = data
        for data_attr in elt_subdomain.findall('Attribute'):
            if data_attr.get('Name') != infoname:
                continue
            icore, data = _get_field(xdmf_file, data_attr.find('DataItem'))
            tra[infoname][ibk][icore] = data
    for info in tra:
        tra[info] = [trab for trab in tra[info] if trab]  # remove empty blocks
        for iblk, trab in enumerate(tra[info]):
            tra[info][iblk] = np.concatenate([trab[icore]
                                              for icore in range(len(trab))])
    return tra


def read_time_h5(h5folder):
    """Iterate through (isnap, istep) recorded in h5folder/'time_botT.h5'.

    Args:
        h5folder (:class:`pathlib.Path`): directory of HDF5 output files.
    Yields:
        tuple of int: (isnap, istep).
    """
    with h5py.File(h5folder / 'time_botT.h5', 'r') as h5f:
        for name, dset in h5f.items():
            yield int(name[-5:]), int(dset[2])
