"""Exceptions raised by StagPy"""


class StagpyError(Exception):

    """Base class for exceptions raised by StagPy"""

    pass


class NoSnapshotError(StagpyError):

    """Raised when no snapshot can be found"""

    def __init__(self, sdat):
        self.sdat = sdat


class NoParFileError(StagpyError):

    """Raised when no par file can be found"""

    def __init__(self, parfile):
        self.parfile = parfile


class NotAvailableError(StagpyError):

    """Raised when a feature is not available yet"""

    pass


class ParsingError(StagpyError):

    """Raised when a parsing error occurs"""

    def __init__(self, faulty_file, msg):
        self.file = faulty_file
        self.msg = msg
