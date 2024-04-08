from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

if typing.TYPE_CHECKING:
    from .parfile import StagyyPar
    from .stagyydata import StagyyData


@dataclass(frozen=True)
class Scales:
    """Dimensional scales."""

    sdat: StagyyData

    @property
    def par(self) -> StagyyPar:
        return self.sdat.par

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
