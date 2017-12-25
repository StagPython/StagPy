"""Exceptions raised by StagPy"""


class StagpyError(Exception):

    """Base class for exceptions raised by StagPy"""

    pass


class NoSnapshotError(StagpyError):

    """Raised when no snapshot can be found"""

    def __init__(self, sdat):
        self.sdat = sdat
