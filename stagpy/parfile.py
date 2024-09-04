"""StagYY par file handling."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import f90nml

from .error import NoParFileError

if typing.TYPE_CHECKING:
    from typing import TypeVar

    from f90nml.namelist import Namelist

    T = TypeVar("T")


@dataclass(frozen=True)
class StagyyPar:
    """A Fortran namelist file written for StagYY."""

    nml: Namelist
    root: Path

    @staticmethod
    def _from_file(parfile: Path) -> StagyyPar:
        if not parfile.is_file():
            raise NoParFileError(parfile)
        par = f90nml.read(str(parfile))
        for section, content in par.items():
            for option, value in content.items():
                try:
                    content[option] = value.strip()
                except AttributeError:
                    pass
        return StagyyPar(nml=par, root=parfile.parent)

    def _update(self, par_new: StagyyPar) -> None:
        for section, content in par_new.nml.items():
            if section in self.nml:
                self.nml[section].update(content)
            else:
                self.nml[section] = content

    @staticmethod
    def from_main_par(parfile: Path, read_parameters_dat: bool = True) -> StagyyPar:
        """Read StagYY namelist `parfile`.

        The namelist is populated in order with:

        - `par_name_defaultparameters` if it is defined in `parfile`;
        - `par_file` itself;
        - `parameters.dat` if it can be found in the StagYY output directories
          and `read_parameters_dat` is `True`.
        """
        par_main = StagyyPar._from_file(parfile)
        if "default_parameters_parfile" in par_main.nml:
            dfltfile = par_main.get(
                "default_parameters_parfile",
                "par_name_defaultparameters",
                "par_defaults",
            )
            par_dflt = StagyyPar._from_file(par_main.root / dfltfile)
            par_dflt._update(par_main)
            par_main = StagyyPar(nml=par_dflt.nml, root=par_main.root)

        if read_parameters_dat:
            outfile = par_main.legacy_output("_parameters.dat")
            if outfile.is_file():
                par_main._update(StagyyPar._from_file(outfile))
            outfile = par_main.h5_output("parameters.dat")
            if outfile.is_file():
                par_main._update(StagyyPar._from_file(outfile))
        return par_main

    def get(self, section: str, option: str, default: T) -> T:
        sec = self.nml.get(section, {})
        return sec.get(option, default)

    def legacy_output(self, suffix: str) -> Path:
        stem = self.get("ioin", "output_file_stem", "test")
        return self.root / (stem + suffix)

    def h5_output(self, filename: str) -> Path:
        h5folder = self.get("ioin", "hdf5_output_folder", "+hdf5")
        return self.root / h5folder / filename
