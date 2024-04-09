"""Types describing StagYY output data."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Varf:
    """Metadata of scalar field.

    `dim` is the dimension used to scale to dimensional values.
    """

    description: str
    dim: str


@dataclass(frozen=True)
class Field:
    """Scalar field and associated metadata."""

    values: NDArray
    meta: Varf


@dataclass(frozen=True)
class Varr:
    """Metadata of radial profiles.

    `dim` is the dimension used to scale to dimensional values.
    """

    description: str
    kind: str
    dim: str


@dataclass(frozen=True)
class Rprof:
    """Radial profile with associated radius and metadata."""

    values: NDArray
    rad: NDArray
    meta: Varr


@dataclass(frozen=True)
class Vart:
    """Metadata of time series.

    `dim` is the dimension used to scale to dimensional values.
    """

    description: str
    kind: str
    dim: str


@dataclass(frozen=True)
class Tseries:
    """A time series with associated time and metadata."""

    values: NDArray
    time: NDArray
    meta: Vart
