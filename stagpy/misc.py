"""miscellaneous definitions"""

from inspect import getdoc
import pathlib
import shutil
import tempfile
from . import conf

INT_FMT = '{:05d}'


def out_name(par_type):
    """return out file name format for any time step"""
    return conf.core.outname + '_' + par_type + INT_FMT


def baredoc(obj):
    """Return the first line of docstring of an object.

    If that first line ends with a period, it is removed in the output.

    Args:
        obj: any Python object with a docstring.

    Returns:
        str: the first line of the docstring of obj.
    """
    doc = getdoc(obj)
    if not doc:
        return ''
    doc = doc.splitlines()[0]
    return doc.rstrip(' .').lstrip()


def fmttime(tin):
    """Time formatting for labels"""
    aaa, bbb = '{:.2e}'.format(tin).split('e')
    bbb = int(bbb)
    return r'$t={} \times 10^{{{}}}$'.format(aaa, bbb)


def list_of_vars(arg_plot):
    """Compute list of variables per plot

    Three nested lists:
    - variables on the same subplots;
    - subplots on the same figure;
    - figures.
    """
    lovs = [[[var for var in svars.split(',') if var]
             for svars in pvars.split('.') if svars]
            for pvars in arg_plot.split('_') if pvars]
    return [lov for lov in lovs if lov]


def set_of_vars(lovs):
    """Build set of variables from list"""
    return set(var for pvars in lovs for svars in pvars for var in svars)


def steps_gen(sdat):
    """Return generator over relevant snapshots or timesteps"""
    if conf.core.snapshots is not None:
        return sdat.snaps[conf.core.snapshots]
    else:
        return sdat.steps[conf.core.timesteps]


def get_rbounds(step):
    """Radii of boundaries"""
    if step.geom is not None:
        rcmb = step.geom.rcmb
    else:
        rcmb = step.sdat.par['geometry']['r_cmb']
        if step.sdat.par['geometry']['shape'].lower() == 'cartesian':
            rcmb = 0
    rcmb = max(rcmb, 0)
    return rcmb, rcmb + 1


class InchoateFiles:

    """Context manager handling files whose names are not known yet"""

    def __init__(self, nfiles=1, tmp_prefix=None):
        """Initialize context object

        nfiles: number of files
        tmp_prefix: prefix name of temporary files
        """
        self._fnames = ['inchoate{}'.format(i) for i in range(nfiles)]
        self._tmpprefix = tmp_prefix
        self._fids = []

    @property
    def fids(self):
        """List of files id"""
        return self._fids

    @property
    def fnames(self):
        """List of filenames"""
        return self._fnames

    @fnames.setter
    def fnames(self, names):
        """Ensure constant size of fnames"""
        names = list(names[:len(self._fnames)])
        self._fnames = names + self._fnames[len(names):]

    def __getitem__(self, idx):
        return self._fids[idx]

    def __enter__(self):
        """Create temporary files"""
        for fname in self.fnames:
            pfx = fname if self._tmpprefix is None else self._tmpprefix
            self._fids.append(
                tempfile.NamedTemporaryFile(
                    mode='w', prefix=pfx, delete=False))
        return self

    def __exit__(self, *exc_info):
        """Give temporary files their final names"""
        for tmp in self._fids:
            tmp.close()
        if exc_info[0] is None:
            for fname, tmp in zip(self.fnames, self._fids):
                shutil.copyfile(tmp.name, fname)
        for tmp in self._fids:
            pathlib.Path(tmp.name).unlink()
