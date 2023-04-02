"""Types describing StagYY output data."""

from __future__ import annotations

import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Varf:
    """Metadata of scalar field.

    Attributes:
        description: short description of the variable.
        dim: dimension used to :func:`~stagpy.stagyydata.StagyyData.scale` to
            dimensional values.
    """

    description: str
    dim: str


@dataclass(frozen=True)
class Field:
    """Scalar field and associated metadata.

    Attributes:
        values: the field itself.
        meta: the metadata of the field.
    """

    values: NDArray
    meta: Varf


@dataclass(frozen=True)
class Varr:
    """Metadata of radial profiles.

    Attributes:
        description: short description of the variable if it is output by
            StagYY, function to compute it otherwise.
        kind: shorter description to group similar variables under the same
            label.
        dim: dimension used to :func:`~stagpy.stagyydata.StagyyData.scale` to
            dimensional values.
    """

    description: str
    kind: str
    dim: str


@dataclass(frozen=True)
class Rprof:
    """Radial profile with associated radius and metadata.

    Attributes:
        values: the profile itself.
        rad: the radial position.
        meta: the metadata of the profile.
    """

    values: NDArray
    rad: NDArray
    meta: Varr


@dataclass(frozen=True)
class Vart:
    """Metadata of time series.

    Attributes:
        description: short description of the variable if it is output by
            StagYY, function to compute it otherwise.
        kind: shorter description to group similar variables under the same
            label.
        dim: dimension used to :func:`~stagpy.stagyydata.StagyyData.scale` to
            dimensional values.
    """

    description: str
    kind: str
    dim: str


@dataclass(frozen=True)
class Tseries:
    """A time series with associated time and metadata.

    Attributes:
        values: the series itself.
        time: the time vector.
        meta: the metadata of the series.
    """

    values: NDArray
    time: NDArray
    meta: Vart
