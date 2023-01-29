"""Various helper functions and classes."""

from __future__ import annotations

import typing
from inspect import getdoc

import matplotlib.pyplot as plt

from . import conf

if typing.TYPE_CHECKING:
    from typing import Any, Optional

    from matplotlib.figure import Figure
    from numpy import ndarray


def out_name(stem: str, timestep: Optional[int] = None) -> str:
    """Return StagPy out file name.

    Args:
        stem: short description of file content.
        timestep: timestep if relevant.

    Returns:
        the output file name.

    Other Parameters:
        conf.core.outname: the generic name stem, defaults to ``'stagpy'``.
    """
    if conf.core.shortname:
        return conf.core.outname
    if timestep is not None:
        stem = f"{stem}{timestep:05d}"
    return conf.core.outname + "_" + stem


def scilabel(value: float, precision: int = 2) -> str:
    """Build scientific notation of some value.

    This is dedicated to use in labels displaying scientific values.

    Args:
        value: numeric value to format.
        precision: number of decimal digits.

    Returns:
        the scientific notation of the specified value.
    """
    man, exps = f"{value:.{precision}e}".split("e")
    exp = int(exps)
    return rf"{man}\times 10^{{{exp}}}"


def saveplot(
    fig: Figure, *name_args: Any, close: bool = True, **name_kwargs: Any
) -> None:
    """Save matplotlib figure.

    You need to provide :data:`stem` as a positional or keyword argument (see
    :func:`out_name`).

    Args:
        fig: the :class:`matplotlib.figure.Figure` to save.
        close: whether to close the figure.
        name_args: positional arguments passed on to :func:`out_name`.
        name_kwargs: keyword arguments passed on to :func:`out_name`.
    """
    oname = out_name(*name_args, **name_kwargs)
    fig.savefig(
        f"{oname}.{conf.plot.format}", format=conf.plot.format, bbox_inches="tight"
    )
    if close:
        plt.close(fig)


def baredoc(obj: object) -> str:
    """Return the first line of the docstring of an object.

    Trailing periods and spaces as well as leading spaces are removed from the
    output.

    Args:
        obj: any Python object.
    Returns:
        str: the first line of the docstring of obj.
    """
    doc = getdoc(obj)
    if not doc:
        return ""
    doc = doc.splitlines()[0]
    return doc.rstrip(" .").lstrip()


def find_in_sorted_arr(value: Any, array: ndarray, after: bool = False) -> int:
    """Return position of element in a sorted array.

    Returns:
        the maximum position i such as array[i] <= value.  If after is True, it
        returns the min i such as value <= array[i] (or 0 if such an index does
        not exist).
    """
    ielt = array.searchsorted(value)
    if ielt == array.size:
        ielt -= 1
    if not after and array[ielt] != value and ielt > 0:
        ielt -= 1
    return ielt
