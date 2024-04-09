from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

from . import phyvars
from .config import Scaling

if typing.TYPE_CHECKING:
    from typing import TypeVar

    from numpy.typing import NDArray

    from .parfile import StagyyPar
    from .stagyydata import StagyyData

    T = TypeVar("T", float, NDArray)


@dataclass(frozen=True)
class Scales:
    """Dimensional scales."""

    sdat: StagyyData

    @property
    def par(self) -> StagyyPar:
        return self.sdat.par

    def make_dimensional(self, data: T, unit: str, scaling: Scaling) -> tuple[T, str]:
        """Scale quantity to obtain dimensional quantity."""
        if self.par.get("switches", "dimensional_units", True) or unit == "1":
            return data, ""
        scale = phyvars.SCALES[unit](self)
        factor = scaling.factors.get(unit, " ")
        if scaling.time_in_y and unit == "s":
            scale /= scaling.yearins
            unit = "yr"
        elif scaling.vel_in_cmpy and unit == "m/s":
            scale *= 100 * scaling.yearins
            unit = "cm/y"
        if factor in phyvars.PREFIXES:
            scale *= 10 ** (-3 * (phyvars.PREFIXES.index(factor) + 1))
            unit = factor + unit
        return data * scale, unit

    @cached_property
    def length(self) -> float:
        """Length in m."""
        thick = self.par.get("geometry", "d_dimensional", 2890e3)
        if self.par.get("boundaries", "air_layer", False):
            thick += self.par.nml["boundaries"]["air_thickness"]
        return thick

    @property
    def temperature(self) -> float:
        """Temperature in K."""
        return self.par.nml["refstate"]["deltaT_dimensional"]

    @property
    def density(self) -> float:
        """Density in kg/m3."""
        return self.par.nml["refstate"]["dens_dimensional"]

    @property
    def th_cond(self) -> float:
        """Thermal conductivity in W/(m.K)."""
        return self.par.nml["refstate"]["tcond_dimensional"]

    @property
    def sp_heat(self) -> float:
        """Specific heat capacity in J/(kg.K)."""
        return self.par.nml["refstate"]["Cp_dimensional"]

    @property
    def dyn_visc(self) -> float:
        """Dynamic viscosity in Pa.s."""
        return self.par.nml["viscosity"]["eta0"]

    @property
    def th_diff(self) -> float:
        """Thermal diffusivity in m2/s."""
        return self.th_cond / (self.density * self.sp_heat)

    @property
    def time(self) -> float:
        """Time in s."""
        return self.length**2 / self.th_diff

    @property
    def velocity(self) -> float:
        """Velocity in m/s."""
        return self.length / self.time

    @property
    def acceleration(self) -> float:
        """Acceleration in m/s2."""
        return self.length / self.time**2

    @property
    def power(self) -> float:
        """Power in W."""
        return self.th_cond * self.temperature * self.length

    @property
    def heat_flux(self) -> float:
        """Local heat flux in W/m2."""
        return self.power / self.length**2

    @property
    def heat_production(self) -> float:
        """Local heat production in W/m3."""
        return self.power / self.length**3

    @property
    def stress(self) -> float:
        """Stress in Pa."""
        return self.dyn_visc / self.time
