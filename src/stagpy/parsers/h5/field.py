from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property

import h5py
import numpy as np

from ...error import ParsingError
from ...phyvars import FIELD, SFIELD
from ._helpers import count_subdomains, ifile_isnap, read_group, try_text
from .xdmf import XmlStream

if typing.TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path
    from typing import Any
    from xml.etree.ElementTree import Element

    from numpy.typing import NDArray


def _make_3d(field: NDArray[np.float64], twod: str | None) -> NDArray[np.float64]:
    """Add a dimension to field if necessary.

    Args:
        field: the field that needs to be 3d.
        twod: 'XZ', 'YZ' or None depending on what is relevant.

    Returns:
        reshaped field.
    """
    shp = list(field.shape)
    if twod and "X" in twod:
        shp.insert(1, 1)
    elif twod:
        shp.insert(0, 1)
    return field.reshape(shp)


def _ncores(
    meshes: list[dict[str, NDArray[np.float64]]], twod: str | None
) -> NDArray[np.float64]:
    """Compute number of nodes in each direction."""
    nnpb = len(meshes)  # number of nodes per block
    nns = [1, 1, 1]  # number of nodes in x, y, z directions
    if twod is None or "X" in twod:
        while (
            nnpb > 1
            and meshes[nns[0]]["X"][0, 0, 0] == meshes[nns[0] - 1]["X"][-1, 0, 0]
        ):
            nns[0] += 1
            nnpb -= 1
    cpu = lambda icy: icy * nns[0]  # noqa: E731
    if twod is None or "Y" in twod:
        while nnpb > 1 and np.isclose(
            meshes[cpu(nns[1])]["Y"][0, 0, 0],
            meshes[cpu(nns[1] - 1)]["Y"][0, -1, 0],
            rtol=1e-10,
        ):
            nns[1] += 1
            nnpb -= nns[0]
    cpu = lambda icz: icz * nns[0] * nns[1]  # noqa: E731
    while nnpb > 1 and np.isclose(
        meshes[cpu(nns[2])]["Z"][0, 0, 0],
        meshes[cpu(nns[2] - 1)]["Z"][0, 0, -1],
        rtol=1e-10,
    ):
        nns[2] += 1
        nnpb -= nns[0] * nns[1]
    assert nnpb == 1
    assert len(meshes) == np.prod(nns)
    return np.array(nns)


def _conglomerate_meshes(
    meshin: list[dict[str, NDArray[np.float64]]], header: dict[str, Any]
) -> dict[str, NDArray[np.float64]]:
    """Conglomerate meshes from several cores into one."""
    meshout = {}
    npc = header["nts"] // header["ncs"]
    shp = [val + 1 if val != 1 else 1 for val in header["nts"]]
    x_p = int(shp[0] != 1)
    y_p = int(shp[1] != 1)
    for coord in meshin[0]:
        meshout[coord] = np.zeros(shp)
    for icore in range(np.prod(header["ncs"])):
        ifs = [
            icore // np.prod(header["ncs"][:i]) % header["ncs"][i] * npc[i]
            for i in range(3)
        ]
        for coord, mesh in meshin[icore].items():
            meshout[coord][
                ifs[0] : ifs[0] + npc[0] + x_p,
                ifs[1] : ifs[1] + npc[1] + y_p,
                ifs[2] : ifs[2] + npc[2] + 1,
            ] = mesh
    return meshout


@dataclass(frozen=True)
class FieldSub:
    file: Path
    dataset: str
    icore: int
    iblock: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class XmfEntry:
    isnap: int
    time: float | None
    mo_lambda: float | None
    mo_thick_sol: float | None
    yin_yang: bool
    twod: str | None
    coord_filepattern: str
    coord_shape: tuple[int, ...]
    range_yin: range
    range_yang: range
    fields: Mapping[str, tuple[int, tuple[int, ...]]]

    def coord_files_yin(self, path_root: Path) -> Iterator[Path]:
        for icore in self.range_yin:
            yield path_root / self.coord_filepattern.format(icore=icore + 1)

    def _fsub(self, path_root: Path, name: str, icore: int, yang: bool) -> FieldSub:
        ifile, shape = self.fields[name]
        yy_tag = "2" if yang else "_"
        ic = icore + 1
        return FieldSub(
            file=path_root / f"{name}_{ifile:05d}_{ic:05d}.h5",
            dataset=f"{name}{yy_tag}{ic:05d}_{self.isnap:05d}",
            icore=icore,
            iblock=int(yang),
            shape=shape,
        )

    def field_subdomains(self, path_root: Path, name: str) -> Iterator[FieldSub]:
        if name not in self.fields:
            return
        for icore in self.range_yin:
            yield self._fsub(path_root, name, icore, False)
        for icore in self.range_yang:
            yield self._fsub(path_root, name, icore, True)


@dataclass(frozen=True)
class FieldXmf:
    path: Path

    def _get_dims(self, elt: Element) -> tuple[int, ...]:
        dims = elt.attrib["Dimensions"].split()
        return tuple(map(int, dims))

    @cached_property
    def _data(self) -> Mapping[int, XmfEntry]:
        xs = XmlStream(filepath=self.path)
        data = {}
        for _ in xs.iter_tag("Time"):
            time = float(xs.current.attrib["Value"])
            xs.advance()
            extra: dict[str, float] = {}
            while xs.current.tag != "Grid":
                # mo_lambda, mo_thick_sol
                extra[xs.current.tag] = float(xs.current.attrib["Value"])
                xs.advance()

            mesh_name = xs.current.attrib["Name"]
            yin_yang = mesh_name.startswith("meshYin")
            i0_yin = int(mesh_name[-5:]) - 1
            twod = None

            xs.skip_to_tag("Geometry")
            with xs.load() as elt_geom:
                if elt_geom.get("Type") == "X_Y":
                    twod = ""
                    for data_item in elt_geom:
                        coord = try_text(xs.filepath, data_item).strip()[-1]
                        if coord in "XYZ":
                            twod += coord
                data_item = elt_geom[0]
                data_text = try_text(xs.filepath, data_item)
                coord_shape = self._get_dims(data_item)
                coord_filepattern = data_text.strip().split(":/", 1)[0]
                coord_file_chunks = coord_filepattern.split("_")
                coord_file_chunks[-2] = "{icore:05d}"
                coord_filepattern = "_".join(coord_file_chunks)

            fields_info = {}
            for elt_fvar in xs.iter_load_successive_tag("Attribute"):
                name = elt_fvar.attrib["Name"]
                elt_data = elt_fvar[0]
                shape = self._get_dims(elt_data)
                ifile, isnap = ifile_isnap(xs.filepath, elt_data)
                fields_info[name] = (ifile, shape)

            r_yin, r_yang = count_subdomains(xs, i0_yin)

            data[isnap] = XmfEntry(
                isnap=isnap,
                time=time,
                mo_lambda=extra.get("mo_lambda"),
                mo_thick_sol=extra.get("mo_thick_sol"),
                yin_yang=yin_yang,
                twod=twod,
                coord_filepattern=coord_filepattern,
                coord_shape=coord_shape,
                range_yin=r_yin,
                range_yang=r_yang,
                fields=fields_info,
            )
        return data

    def __getitem__(self, isnap: int) -> XmfEntry:
        try:
            return self._data[isnap]
        except KeyError:
            raise ParsingError(self.path, f"no data for snapshot {isnap}")


def read_geom(xdmf: FieldXmf, snapshot: int) -> dict[str, Any]:
    """Extract geometry information from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        snapshot: snapshot number.

    Returns:
        geometry information.
    """
    header: dict[str, Any] = {}

    entry = xdmf[snapshot]
    header["ti_ad"] = entry.time
    header["mo_lambda"] = entry.mo_lambda
    header["mo_thick_sol"] = entry.mo_thick_sol
    header["ntb"] = 2 if entry.yin_yang else 1

    all_meshes: list[dict[str, NDArray[np.float64]]] = []
    for h5file in entry.coord_files_yin(xdmf.path.parent):
        all_meshes.append({})
        with h5py.File(h5file, "r") as h5f:
            for coord, mesh in h5f.items():
                # for some reason, the array is transposed!
                all_meshes[-1][coord] = mesh[()].reshape(entry.coord_shape).T
                all_meshes[-1][coord] = _make_3d(all_meshes[-1][coord], entry.twod)

    header["ncs"] = _ncores(all_meshes, entry.twod)
    header["nts"] = list(
        (all_meshes[0]["X"].shape[i] - 1) * header["ncs"][i] for i in range(3)
    )
    header["nts"] = np.array([max(1, val) for val in header["nts"]])
    meshes = _conglomerate_meshes(all_meshes, header)
    if np.any(meshes["Z"][:, :, 0] != 0):
        # spherical
        if entry.twod is not None:  # annulus geometry...
            header["x_mesh"] = np.copy(meshes["Y"])
            header["y_mesh"] = np.copy(meshes["Z"])
            header["z_mesh"] = np.copy(meshes["X"])
        else:  # YinYang, here only yin
            header["x_mesh"] = np.copy(meshes["X"])
            header["y_mesh"] = np.copy(meshes["Y"])
            header["z_mesh"] = np.copy(meshes["Z"])
        header["r_mesh"] = np.sqrt(
            header["x_mesh"] ** 2 + header["y_mesh"] ** 2 + header["z_mesh"] ** 2
        )
        header["t_mesh"] = np.arccos(header["z_mesh"] / header["r_mesh"])
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], -header["x_mesh"]) + np.pi, -1, 1
        )
        header["e1_coord"] = header["t_mesh"][:, 0, 0]
        header["e2_coord"] = header["p_mesh"][0, :, 0]
        header["e3_coord"] = header["r_mesh"][0, 0, :]
    else:
        header["e1_coord"] = meshes["X"][:, 0, 0]
        header["e2_coord"] = meshes["Y"][0, :, 0]
        header["e3_coord"] = meshes["Z"][0, 0, :]
    header["aspect"] = (
        header["e1_coord"][-1] - header["e2_coord"][0],
        header["e1_coord"][-1] - header["e2_coord"][0],
    )
    header["rcmb"] = header["e3_coord"][0]
    if header["rcmb"] == 0:
        header["rcmb"] = -1
    else:
        header["e3_coord"] = header["e3_coord"] - header["rcmb"]
    if entry.twod is None or "X" in entry.twod:
        header["e1_coord"] = header["e1_coord"][:-1]
    if entry.twod is None or "Y" in entry.twod:
        header["e2_coord"] = header["e2_coord"][:-1]
    header["e3_coord"] = header["e3_coord"][:-1]

    return header


def _to_spherical(
    flds: NDArray[np.float64], header: dict[str, Any]
) -> NDArray[np.float64]:
    """Convert vector field to spherical."""
    cth = np.cos(header["t_mesh"][:, :, :-1])
    sth = np.sin(header["t_mesh"][:, :, :-1])
    cph = np.cos(header["p_mesh"][:, :, :-1])
    sph = np.sin(header["p_mesh"][:, :, :-1])
    fout = np.copy(flds)
    fout[0] = cth * cph * flds[0] + cth * sph * flds[1] - sth * flds[2]
    fout[1] = sph * flds[0] - cph * flds[1]  # need to take the opposite here
    fout[2] = sth * cph * flds[0] + sth * sph * flds[1] + cth * flds[2]
    return fout


def _flds_shape(vector_field: bool, header: dict[str, Any]) -> list[int]:
    """Compute shape of flds variable."""
    shp = list(header["nts"])
    shp.append(header["ntb"])
    if vector_field:
        shp.insert(0, 3)
        # extra points
        header["xp"] = int(header["nts"][0] != 1)
        shp[1] += header["xp"]
        header["yp"] = int(header["nts"][1] != 1)
        shp[2] += header["yp"]
        header["zp"] = 1
        header["xyp"] = 1
    else:
        shp.insert(0, 1)
        header["xp"] = 0
        header["yp"] = 0
        header["zp"] = 0
        header["xyp"] = 0
    return shp


def _post_read_flds(
    flds: NDArray[np.float64], header: dict[str, Any]
) -> NDArray[np.float64]:
    """Process flds to handle sphericity."""
    if flds.shape[0] >= 3 and header["rcmb"] > 0:
        # spherical vector
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], header["x_mesh"]), -1, 1
        )
        for ibk in range(header["ntb"]):
            flds[..., ibk] = _to_spherical(flds[..., ibk], header)
        header["p_mesh"] = np.roll(
            np.arctan2(header["y_mesh"], -header["x_mesh"]) + np.pi, -1, 1
        )
    return flds


def field(
    xdmf: FieldXmf,
    fieldname: str,
    snapshot: int,
    header: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], NDArray[np.float64]] | None:
    """Extract field data from hdf5 files.

    Args:
        xdmf: xdmf file parser.
        fieldname: name of field to extract.
        snapshot: snapshot number.
        header: geometry information.

    Returns:
        geometry information and field data. None is returned if data is
            unavailable.
    """
    if header is None:
        header = read_geom(xdmf, snapshot)

    vector_field = len(FIELD.h5_files.get(fieldname, [])) == 3
    surface_field = fieldname in SFIELD.h5_files

    npc = header["nts"] // header["ncs"]  # number of grid point per node
    flds = np.zeros(_flds_shape(vector_field, header))
    data_found = False

    for fsub in xdmf[snapshot].field_subdomains(xdmf.path.parent, fieldname):
        fld = read_group(fsub.file, fsub.dataset).reshape(fsub.shape)
        # for some reason, the field is transposed
        fld = fld.T
        shp = fld.shape

        if shp[-1] == 1 and header["nts"][0] == 1:  # YZ
            fld = fld.reshape((shp[0], 1, shp[1], shp[2]))
            if header["rcmb"] < 0:
                fld = fld[(2, 0, 1), ...]
        elif shp[-1] == 1:  # XZ
            fld = fld.reshape((shp[0], shp[1], 1, shp[2]))
            if header["rcmb"] < 0:
                fld = fld[(1, 2, 0), ...]
        elif surface_field:
            fld = fld.reshape((1, npc[0], npc[1], 1))
        elif header["nts"][1] == 1:  # cart XZ
            fld = fld.reshape((1, shp[0], 1, shp[1]))
        ifs = [
            fsub.icore // np.prod(header["ncs"][:i]) % header["ncs"][i] * npc[i]
            for i in range(3)
        ]
        if surface_field:
            ifs[2] = 0
            npc[2] = 1
        if header["zp"]:  # remove top row
            fld = fld[:, :, :, :-1]
        flds[
            :,
            ifs[0] : ifs[0] + npc[0] + header["xp"],
            ifs[1] : ifs[1] + npc[1] + header["yp"],
            ifs[2] : ifs[2] + npc[2],
            fsub.iblock,
        ] = fld
        data_found = True

    if flds.shape[0] == 3 and flds.shape[-1] == 2:  # YinYang vector
        # Yang grid is rotated compared to Yin grid
        flds[0, ..., 1] = -flds[0, ..., 1]
        vt = flds[1, ..., 1].copy()
        flds[1, ..., 1] = flds[2, ..., 1]
        flds[2, ..., 1] = vt
    flds = _post_read_flds(flds, header)

    if surface_field:
        # remove z component
        flds = flds[..., 0, :]
    return (header, flds) if data_found else None
