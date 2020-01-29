"""Miscellaneous definitions."""

from inspect import getdoc
import pathlib
import shutil
import tempfile

import matplotlib.pyplot as plt

from . import conf

INT_FMT = '{:05d}'


def out_name(stem, timestep=None):
    """Return StagPy out file name.

    Args:
        stem (str): short description of file content.
        timestep (int): timestep if relevant.

    Returns:
        str: the output file name.

    Other Parameters:
        conf.core.outname (str): the generic name stem, defaults to
            ``'stagpy'``.
    """
    if conf.core.shortname:
        return conf.core.outname
    if timestep is not None:
        stem = (stem + INT_FMT).format(timestep)
    return conf.core.outname + '_' + stem


def scilabel(value, precision=2):
    """Build scientific notation of some value.

    This is dedicated to use in labels displaying scientific values.

    Args:
        value (float): numeric value to format.
        precision (int): number of decimal digits.

    Returns:
        str: the scientific notation the specified value.
    """
    fmt = '{{:.{}e}}'.format(precision)
    man, exp = fmt.format(value).split('e')
    exp = int(exp)
    return r'{}\times 10^{{{}}}'.format(man, exp)


def saveplot(fig, *name_args, close=True, **name_kwargs):
    """Save matplotlib figure.

    You need to provide :data:`stem` as a positional or keyword argument (see
    :func:`out_name`).

    Args:
        fig (:class:`matplotlib.figure.Figure`): matplotlib figure.
        close (bool): whether to close the figure.
        name_args: positional arguments passed on to :func:`out_name`.
        name_kwargs: keyword arguments passed on to :func:`out_name`.
    """
    oname = out_name(*name_args, **name_kwargs)
    fig.savefig('{}.{}'.format(oname, conf.plot.format),
                format=conf.plot.format, bbox_inches='tight')
    if close:
        plt.close(fig)


def baredoc(obj):
    """Return the first line of the docstring of an object.

    Trailing periods and spaces as well as leading spaces are removed from the
    output.

    Args:
        obj: any Python object.
    Returns:
        str: the first line of the docstring of obj.
    """
    doc = getdoc(obj)
    if not doc:
        return ''
    doc = doc.splitlines()[0]
    return doc.rstrip(' .').lstrip()


def list_of_vars(arg_plot):
    """Construct list of variables per plot.

    Args:
        arg_plot (str): string with variable names separated with
            ``-`` (figures), ``.`` (subplots) and ``,`` (same subplot).
    Returns:
        three nested lists of str

        - variables on the same subplot;
        - subplots on the same figure;
        - figures.
    """
    lovs = [[[var for var in svars.split(',') if var]
             for svars in pvars.split('.') if svars]
            for pvars in arg_plot.split('-') if pvars]
    lovs = [[slov for slov in lov if slov] for lov in lovs if lov]
    return [lov for lov in lovs if lov]


def set_of_vars(lovs):
    """Build set of variables from list.

    Args:
        lovs: nested lists of variables such as the one produced by
            :func:`list_of_vars`.
    Returns:
        set of str: flattened set of all the variables present in the
        nested lists.
    """
    return set(var for pvars in lovs for svars in pvars for var in svars)


def get_rbounds(step):
    """Radial or vertical position of boundaries.

    Args:
        step (:class:`~stagpy.stagyydata._Step`): a step of a StagyyData
            instance.
    Returns:
        tuple of floats: radial or vertical positions of boundaries of the
        domain.
    """
    if step.geom is not None:
        rcmb = step.geom.rcmb
    else:
        rcmb = step.sdat.par['geometry']['r_cmb']
        if step.sdat.par['geometry']['shape'].lower() == 'cartesian':
            rcmb = 0
    rcmb = max(rcmb, 0)
    return rcmb, rcmb + 1


class InchoateFiles:
    """Context manager handling files whose names are not known yet.

    Example:
        InchoateFiles is used here to manage three files::

            with InchoateFiles(3) as incho:
                # for convenience, incho[x] is the same as incho.fids[x]
                incho[0].write('First file')
                incho[1].write('Second file')
                incho[2].write('Third file')

                # the three files will be named 'tata', 'titi' and 'toto'
                incho.fnames = ['tata', 'titi', 'toto']

    Args:
        nfiles (int): number of files. Defaults to 1.
        tmp_prefix (str): prefix name of temporary files. Use this
            parameter if you want to easily track down the temporary files
            created by the manager.
    """

    def __init__(self, nfiles=1, tmp_prefix=None):
        self._fnames = ['inchoate{}'.format(i) for i in range(nfiles)]
        self._tmpprefix = tmp_prefix
        self._fids = []

    @property
    def fids(self):
        """List of files id.

        Use this to perform operations on files when the context manager is
        used. :meth:`InchoateFiles.__getitem__` is implemented in order to
        provide direct access to this property content (``self[x]`` is the
        same as ``self.fids[x]``).
        """
        return self._fids

    @property
    def fnames(self):
        """List of filenames.

        Set this to the list of final filenames before exiting the context
        manager. If this list is not set by the user, the produced files will
        be named ``'inchoateN'`` with ``N`` the index of the file. If the list
        of names you set is too long, it will be truncated. If it is too short,
        extra files will be named ``'inchoateN'``.
        """
        return self._fnames

    @fnames.setter
    def fnames(self, names):
        """Ensure constant size of fnames."""
        names = list(names[:len(self._fnames)])
        self._fnames = names + self._fnames[len(names):]

    def __getitem__(self, idx):
        return self._fids[idx]

    def __enter__(self):
        """Create temporary files."""
        for fname in self.fnames:
            pfx = fname if self._tmpprefix is None else self._tmpprefix
            self._fids.append(
                tempfile.NamedTemporaryFile(
                    mode='w', prefix=pfx, delete=False))
        return self

    def __exit__(self, *exc_info):
        """Give temporary files their final names."""
        for tmp in self._fids:
            tmp.close()
        if exc_info[0] is None:
            for fname, tmp in zip(self.fnames, self._fids):
                shutil.copyfile(tmp.name, fname)
        for tmp in self._fids:
            pathlib.Path(tmp.name).unlink()
