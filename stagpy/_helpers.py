"""Various helper functions and classes."""

from __future__ import annotations
from inspect import getdoc
from typing import TYPE_CHECKING, Generic, TypeVar

import matplotlib.pyplot as plt

from . import conf

if TYPE_CHECKING:
    from typing import Optional, Any, List, Callable, NoReturn
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
        stem = f'{stem}{timestep:05d}'
    return conf.core.outname + '_' + stem


def scilabel(value: float, precision: int = 2) -> str:
    """Build scientific notation of some value.

    This is dedicated to use in labels displaying scientific values.

    Args:
        value: numeric value to format.
        precision: number of decimal digits.

    Returns:
        the scientific notation of the specified value.
    """
    man, exps = f'{value:.{precision}e}'.split('e')
    exp = int(exps)
    return fr'{man}\times 10^{{{exp}}}'


def saveplot(fig: Figure, *name_args: Any, close: bool = True,
             **name_kwargs: Any) -> None:
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
    fig.savefig(f'{oname}.{conf.plot.format}',
                format=conf.plot.format, bbox_inches='tight')
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
        return ''
    doc = doc.splitlines()[0]
    return doc.rstrip(' .').lstrip()


def list_of_vars(arg_plot: str) -> List[List[List[str]]]:
    """Construct list of variables per plot.

    Args:
        arg_plot: variable names separated with ``-`` (figures),
            ``.`` (subplots) and ``,`` (same subplot).
    Returns:
        three nested lists of str

        - variables on the same subplot;
        - subplots on the same figure;
        - figures.
    """
    lovs = [[[var for var in svars.split(',') if var]
             for svars in pvars.split('.') if svars]
            for pvars in arg_plot.split('-') if pvars]
    lovs = [[slov for slov in lov if slov] for lov in lovs if lov]
    return [lov for lov in lovs if lov]


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


T = TypeVar('T')
V = TypeVar('V')


class CachedReadOnlyProperty(Generic[T, V]):
    """Descriptor implementation of read-only cached properties.

    Properties are cached as ``_cropped_{name}`` instance attribute.

    This is preferable to using a combination of ``@property`` and
    ``@functools.lru_cache`` since the cache is bound to instances and
    therefore get GCd with the instance when the latter is no longer in use
    instead of staying in the cache which would use the instance itself as its
    key.

    This also has an advantage over ``@cached_property`` (Python>3.8): the
    property is read-only instead of being writeable.
    """

    def __init__(self, thunk: Callable[[T], V]):
        self._thunk = thunk
        self._name = thunk.__name__
        self._cache_name = f'_cropped_{self._name}'
        self.__doc__ = thunk.__doc__

    def __get__(self, instance: T, _: Any) -> V:
        try:
            return getattr(instance, self._cache_name)
        except AttributeError:
            pass
        cached_value = self._thunk(instance)
        setattr(instance, self._cache_name, cached_value)
        return cached_value

    def __set__(self, instance: T, _: Any) -> NoReturn:
        raise AttributeError(
            f'Cannot set {self._name} property of {instance!r}')
