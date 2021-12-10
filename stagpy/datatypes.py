"""Types describing StagYY output data."""

from __future__ import annotations
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import ndarray


class Varf(NamedTuple):
    """Metadata of scalar field.

    Attributes:
        description: short description of the variable.
        dim: dimension used to :func:`~stagpy.stagyydata.StagyyData.scale` to
            dimensional values.
    """

    description: str
    dim: str


class Field(NamedTuple):
    """Scalar field and associated metadata.

    Attributes:
        values: the field itself.
        meta: the metadata of the field.
    """

    values: ndarray
    meta: Varf


class Varr(NamedTuple):
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
    # Callable[[Step], Tuple[ndarray, ndarray]]]
    kind: str
    dim: str


class Rprof(NamedTuple):
    """Radial profile with associated radius and metadata.

    Attributes:
        values: the profile itself.
        rad: the radial position.
        meta: the metadata of the profile.
    """

    values: ndarray
    rad: ndarray
    meta: Varr


class Vart(NamedTuple):
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


class Tseries(NamedTuple):
    """A time series with associated time and metadata.

    Attributes:
        values: the series itself.
        time: the time vector.
        meta: the metadata of the series.
    """

    values: ndarray
    time: ndarray
    meta: Vart
