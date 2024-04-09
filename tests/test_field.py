import pytest

import stagpy.error
import stagpy.phyvars
from stagpy.config import Config
from stagpy.field import get_meshes_fld, get_meshes_vec, valid_field_var
from stagpy.stagyydata import Step


def test_field_unknown(step: Step) -> None:
    with pytest.raises(stagpy.error.UnknownFieldVarError):
        step.fields["InvalidField"]
        assert "InvalidField" in step.fields


def test_field_missing(step: Step) -> None:
    with pytest.raises(stagpy.error.MissingDataError):
        step.fields["rsc"]
    assert "rsc" not in step.fields


def test_valid_field_var() -> None:
    for var in stagpy.phyvars.FIELD:
        assert valid_field_var(var)
    for var in stagpy.phyvars.FIELD_EXTRA:
        assert valid_field_var(var)


def test_valid_field_var_invalid() -> None:
    assert not valid_field_var("dummyfieldvar")


def test_get_meshes_fld_no_walls(step: Step) -> None:
    xmesh, ymesh, fld, meta = get_meshes_fld(Config.default_(), step, "T", walls=False)
    assert len(fld.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == fld.shape[0]
    assert xmesh.shape[1] == ymesh.shape[1] == fld.shape[1]
    assert meta.description == "Temperature"


def test_get_meshes_fld_walls(step: Step) -> None:
    xmesh, ymesh, fld, meta = get_meshes_fld(Config.default_(), step, "T", walls=True)
    assert len(fld.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == fld.shape[0] + 1
    assert xmesh.shape[1] == ymesh.shape[1] == fld.shape[1] + 1
    assert meta.description == "Temperature"


def test_get_meshes_vec(step: Step) -> None:
    xmesh, ymesh, vec1, vec2 = get_meshes_vec(Config.default_(), step, "v")
    assert len(vec1.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == vec1.shape[0] == vec2.shape[0]
    assert xmesh.shape[1] == ymesh.shape[1] == vec1.shape[1] == vec2.shape[1]
