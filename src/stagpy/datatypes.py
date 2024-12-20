"""Types describing StagYY output data."""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Varf:
    """Metadata of scalar field."""

    description: str
    """description of field variable."""
    dim: str
    """dimension used to scale to dimensional values."""


@dataclass(frozen=True)
class Field:
    """Scalar field and associated metadata."""

    values: NDArray[np.float64]
    """values of field."""
    meta: Varf
    """metadata."""


@dataclass(frozen=True)
class Varr:
    """Metadata of radial profiles."""

    description: str
    """description of profile variable."""
    kind: str
    """short description"""
    dim: str
    """dimension used to scale to dimensional values."""


@dataclass(frozen=True)
class Rprof:
    """Radial profile with associated radius and metadata."""

    values: NDArray[np.float64]
    """values of profile."""
    rad: NDArray[np.float64]
    """radial position of profile."""
    meta: Varr
    """metadata."""


@dataclass(frozen=True)
class Vart:
    """Metadata of time series."""

    description: str
    """description of time series."""
    kind: str
    """short description"""
    dim: str
    """dimension used to scale to dimensional values."""


@dataclass(frozen=True)
class Tseries:
    """A time series with associated time and metadata."""

    values: NDArray[np.float64]
    """values of time series."""
    time: NDArray[np.float64]
    """time position of time series."""
    meta: Vart
    """metadata."""
