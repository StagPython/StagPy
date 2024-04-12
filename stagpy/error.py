"""Exceptions raised by StagPy."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from os import PathLike

    from .stagyydata import StagyyData
    from .step import Step


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

    Attributes:
        sdat (StagyyData): the `StagyyData` for which no snapshot was found.
    """

    def __init__(self, sdat: StagyyData):
        self.sdat = sdat
        super().__init__(f"no snapshot found for {sdat}")


class NoGeomError(StagpyError):
    """Raised when no geometry info can be found.

    Attributes:
        step (Step): the `Step` for which no geometry was found.
    """

    def __init__(self, step: Step):
        self.step = step
        super().__init__(f"no geometry info found for {step!r}")


class NoTimeError(StagpyError):
    """Raised when no time can be found for a step.

    Attributes:
        step (Step): the `Step` instance for which no geometry was found.
    """

    def __init__(self, step: Step):
        self.step = step
        super().__init__(f"no time found for {step!r}")


class NoRefstateError(StagpyError):
    """Raised when no refstate output can be found.

    Attributes:
        sdat (StagyyData): the `StagyyData` instance for which no refstate was found.
    """

    def __init__(self, sdat: StagyyData):
        self.sdat = sdat
        super().__init__(f"no refstate found for {sdat!r}")


class NoParFileError(StagpyError):
    """Raised when no par file can be found.

    Attributes:
        parfile (PathLike): the expected path of the par file.
    """

    def __init__(self, parfile: PathLike):
        self.parfile = parfile
        super().__init__(f"{parfile} file not found")


class NotAvailableError(StagpyError):
    """Raised when a feature is not available yet."""

    pass


class ParsingError(StagpyError):
    """Raised when a parsing error occurs.

    Attributes:
        file (PathLike): path of the file where a parsing problem was encountered.
        msg (str): error message.
    """

    def __init__(self, file: PathLike, msg: str):
        self.file = file
        self.msg = msg
        super().__init__(file, msg)


class InvalidTimestepError(StagpyError, KeyError):
    """Raised when invalid time step is requested.

    Attributes:
        sdat (StagyyData): the `StagyyData` instance to which the request was made.
        istep (int): the invalid time step index.
        msg (str): the error message.
    """

    def __init__(self, sdat: StagyyData, istep: int, msg: str):
        self.sdat = sdat
        self.istep = istep
        self.msg = msg
        super().__init__(sdat, istep, msg)


class InvalidSnapshotError(StagpyError, KeyError):
    """Raised when invalid snapshot is requested.

    Attributes:
        sdat (StagyyData): the `StagyyData` instance to which the request was made.
        isnap (int): the invalid snapshot index.
        msg (str): the error message.
    """

    def __init__(self, sdat: StagyyData, isnap: int, msg: str):
        self.sdat = sdat
        self.isnap = isnap
        self.msg = msg
        super().__init__(sdat, isnap, msg)


class InvalidTimeFractionError(StagpyError):
    """Raised when invalid fraction of series is requested.

    Attributes:
        fraction (float): the invalid fraction.
    """

    def __init__(self, fraction: float):
        self.fraction = fraction
        super().__init__(f"Fraction should be in (0,1] (received {fraction})")


class InvalidNfieldsError(StagpyError):
    """Raised when invalid nfields_max is requested.

    Attributes:
        nfields (int): the invalid number of field.
    """

    def __init__(self, nfields: int):
        self.nfields = nfields
        super().__init__(f"nfields_max should be >5 (received {nfields})")


class InvalidZoomError(StagpyError):
    """Raised when invalid zoom is requested.

    Attributes:
        zoom (float): the invalid zoom level.
    """

    def __init__(self, zoom: float):
        self.zoom = zoom
        super().__init__(f"Zoom angle should be in [0,360] (received {zoom})")


class MissingDataError(StagpyError, KeyError):
    """Raised when requested data is not present in output."""

    pass


class UnknownVarError(StagpyError, KeyError):
    """Raised when invalid var is requested.

    Attributes:
        varname (str): the invalid var name.
    """

    def __init__(self, varname: str):
        self.varname = varname
        super().__init__(varname)


class UnknownFieldVarError(UnknownVarError):
    """Raised when invalid field var is requested."""

    pass


class UnknownRprofVarError(UnknownVarError):
    """Raised when invalid rprof var is requested."""

    pass


class UnknownTimeVarError(UnknownVarError):
    """Raised when invalid time var is requested."""

    pass
