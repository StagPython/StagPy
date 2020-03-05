"""Exceptions raised by StagPy."""


class StagpyError(Exception):
    """Base class for exceptions raised by StagPy.

    Note:
        All exceptions derive from this class. To catch any error that might be
        raised by StagPy due to invalid requests/missing data, you only need to
        catch this exception.
    """

    pass


class NoSnapshotError(StagpyError):
    """Raised when no snapshot can be found.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which no snapshot were found.

    Attributes:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which no snapshot were found.
    """

    def __init__(self, sdat):
        self.sdat = sdat
        super().__init__('no snapshot found for {}'.format(sdat))


class NoParFileError(StagpyError):
    """Raised when no par file can be found.

    Args:
        parfile (pathlike): the expected path of
            the par file.

    Attributes:
        parfile (pathlike): the expected path of the par file.
    """

    def __init__(self, parfile):
        self.parfile = parfile
        super().__init__('{} file not found'.format(parfile))


class NotAvailableError(StagpyError):
    """Raised when a feature is not available yet."""

    pass


class ParsingError(StagpyError):
    """Raised when a parsing error occurs.

    Args:
        faulty_file (pathlike): path of the file where a parsing problem
            was encountered.
        msg (str): error message.

    Attributes:
        file (pathlike): path of the file where a parsing problem was
            encountered.
        msg (str): error message.
    """

    def __init__(self, faulty_file, msg):
        self.file = faulty_file
        self.msg = msg
        super().__init__(faulty_file, msg)


class InvalidTimestepError(StagpyError, KeyError):
    """Raised when invalid time step is requested.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which the request was made.
        istep (int): the invalid time step index.
        msg (str): the error message.

    Attributes:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which the request was made.
        istep (int): the invalid time step index.
        msg (str): the error message.
    """

    def __init__(self, sdat, istep, msg):
        self.sdat = sdat
        self.istep = istep
        self.msg = msg
        super().__init__(sdat, istep, msg)


class InvalidSnapshotError(StagpyError, KeyError):
    """Raised when invalid snapshot is requested.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which the request was made.
        isnap (int): the invalid snapshot index.
        msg (str): the error message.

    Attributes:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which the request was made.
        isnap (int): the invalid snapshot index.
        msg (str): the error message.
    """

    def __init__(self, sdat, isnap, msg):
        self.sdat = sdat
        self.isnap = isnap
        self.msg = msg
        super().__init__(sdat, isnap, msg)


class InvalidTimeFractionError(StagpyError):
    """Raised when invalid fraction of series is requested.

    Args:
        fraction (float): the invalid fraction.

    Attributes:
        fraction (float): the invalid fraction.
    """

    def __init__(self, fraction):
        self.fraction = fraction
        super().__init__('Fraction should be in (0,1] (received {})'
                         .format(fraction))


class InvalidNfieldsError(StagpyError):
    """Raised when invalid nfields_max is requested.

    Args:
        nfields (int): the invalid number of fields.

    Attributes:
        nfields (int): the invalid number of field.
    """

    def __init__(self, nfields):
        self.nfields = nfields
        super().__init__('nfields_max should be >5 (received {})'
                         .format(nfields))


class InvalidZoomError(StagpyError):
    """Raised when invalid zoom is requested.

    Args:
        zoom (int): the invalid zoom level.

    Attributes:
        zoom (int): the invalid zoom level.
    """

    def __init__(self, zoom):
        self.zoom = zoom
        super().__init__('Zoom angle should be in [0,360] (received {})'
                         .format(zoom))


class StagnantLidError(StagpyError):
    """Raised when unexpected stagnant lid regime is found.

    Args:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which a stagnant lid regime was found.

    Attributes:
        sdat (:class:`~stagpy.stagyydata.StagyyData`): the StagyyData
            instance for which a stagnant lid regime was found.
    """

    def __init__(self, sdat):
        self.sdat = sdat
        super().__init__('Stagnant lid regime for {}'.format(sdat))


class MissingDataError(StagpyError, KeyError):
    """Raised when requested data is not present in output."""

    pass


class UnknownVarError(StagpyError, KeyError):
    """Raised when invalid var is requested.

    Args:
        varname (str): the invalid var name.

    Attributes:
        varname (str): the invalid var name.
    """

    def __init__(self, varname):
        self.varname = varname
        super().__init__(varname)


class UnknownFiltersError(StagpyError):
    """Raised when invalid step filter is requested.

    Args:
        filters (list): the invalid filter names.

    Attributes:
        filters (list): the invalid filter names.
    """

    def __init__(self, filters):
        self.filters = filters
        super().__init__(', '.join(repr(f) for f in filters))


class UnknownFieldVarError(UnknownVarError):
    """Raised when invalid field var is requested.

    Derived from :class:`~stagpy.error.UnknownVarError`.
    """

    pass


class UnknownRprofVarError(UnknownVarError):
    """Raised when invalid rprof var is requested.

    Derived from :class:`~stagpy.error.UnknownVarError`.
    """

    pass


class UnknownTimeVarError(UnknownVarError):
    """Raised when invalid time var is requested.

    Derived from :class:`~stagpy.error.UnknownVarError`.
    """

    pass
