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
