"""Exceptions raised by StagPy"""


class StagpyError(Exception):

    """Base class for exceptions raised by StagPy"""

    pass


class NoSnapshotError(StagpyError):

    """Raised when no snapshot can be found"""

    def __init__(self, sdat):
        self.sdat = sdat
        super().__init__('no snapshot found for {}'.format(sdat))


class NoParFileError(StagpyError):

    """Raised when no par file can be found"""

    def __init__(self, parfile):
        self.parfile = parfile
        super().__init__('{} file not found'.format(parfile))


class NotAvailableError(StagpyError):

    """Raised when a feature is not available yet"""

    pass


class ParsingError(StagpyError):

    """Raised when a parsing error occurs"""

    def __init__(self, faulty_file, msg):
        self.file = faulty_file
        self.msg = msg
        super().__init__(faulty_file, msg)


class InvalidTimestepError(StagpyError):

    """Raised when invalid time step is requested"""

    def __init__(self, sdat, istep, msg):
        self.sdat = sdat
        self.istep = istep
        self.msg = msg
        super().__init__(sdat, istep, msg)


class UnknownVarError(StagpyError):

    """Raised when invalid var is requested"""

    def __init__(self, varname):
        self.varname = varname
        super().__init__(varname)


class UnknownFieldVarError(UnknownVarError):

    """Raised when invalid field var is requested"""

    pass


class UnknownRprofVarError(UnknownVarError):

    """Raised when invalid rprof var is requested"""

    pass


class UnknownTimeVarError(UnknownVarError):

    """Raised when invalid time var is requested"""

    pass
