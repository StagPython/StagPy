"""Exceptions raised by StagPy."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

    from .stagyydata import StagyyData
    from .step import Step


class StagpyError(Exception):
    """Base class for exceptions raised by StagPy.

    Note:
        All exceptions derive from this class. To catch any error that might be
        raised by StagPy due to invalid requests/missing data, you only need to
        catch this exception.
    """


@dataclass
class NoSnapshotError(StagpyError):
    """Raised when no snapshot can be found."""

    sdat: StagyyData


@dataclass
class NoGeomError(StagpyError):
    """Raised when no geometry info can be found."""

    step: Step


@dataclass
class NoTimeError(StagpyError):
    """Raised when no time can be found for a step."""

    step: Step


@dataclass
class NoRefstateError(StagpyError):
    """Raised when no refstate output can be found."""

    sdat: StagyyData


@dataclass
class NoParFileError(StagpyError):
    """Raised when no par file can be found."""

    parfile: Path


class NotAvailableError(StagpyError):
    """Raised when a feature is not available yet."""


@dataclass
class ParsingError(StagpyError):
    """Raised when a parsing error occurs."""

    file: Path
    msg: str


@dataclass
class InvalidTimestepError(StagpyError):
    """Raised when invalid time step is requested."""

    sdat: StagyyData
    istep: int
    msg: str


@dataclass
class InvalidSnapshotError(StagpyError):
    """Raised when invalid snapshot is requested."""

    sdat: StagyyData
    isnap: int
    msg: str


@dataclass
class InvalidTimeFractionError(StagpyError):
    """Raised when invalid fraction of series is requested, should be in (0, 1]."""

    fraction: float


@dataclass
class InvalidNfieldsError(StagpyError):
    """Raised when invalid nfields_max is requested."""

    nfields: int


@dataclass
class InvalidZoomError(StagpyError):
    """Raised when invalid zoom is requested, should be in [0, 360]."""

    zoom: float


class MissingDataError(StagpyError):
    """Raised when requested data is not present in output."""


@dataclass
class UnknownVarError(StagpyError):
    """Raised when invalid var is requested."""

    varname: str


class UnknownFieldVarError(UnknownVarError):
    """Raised when invalid field var is requested."""


class UnknownRprofVarError(UnknownVarError):
    """Raised when invalid rprof var is requested."""


class UnknownTimeVarError(UnknownVarError):
    """Raised when invalid time var is requested."""
