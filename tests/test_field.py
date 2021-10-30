import pytest
import stagpy.error
import stagpy.field
import stagpy.phyvars


def test_field_unknown(step):
    with pytest.raises(stagpy.error.UnknownFieldVarError):
        step.fields['InvalidField']
        assert 'InvalidField' in step.fields


def test_field_missing(step):
    with pytest.raises(stagpy.error.MissingDataError):
        step.fields['rsc']
    assert 'rsc' not in step.fields


def test_valid_field_var():
    for var in stagpy.phyvars.FIELD:
        assert stagpy.field.valid_field_var(var)
    for var in stagpy.phyvars.FIELD_EXTRA:
        assert stagpy.field.valid_field_var(var)


def test_valid_field_var_invalid():
    assert not stagpy.field.valid_field_var('dummyfieldvar')


def test_get_meshes_fld_no_walls(step):
    xmesh, ymesh, fld, meta = stagpy.field.get_meshes_fld(step, 'T',
                                                          walls=False)
    assert len(fld.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == fld.shape[0]
    assert xmesh.shape[1] == ymesh.shape[1] == fld.shape[1]
    assert meta.description == 'Temperature'


def test_get_meshes_fld_walls(step):
    xmesh, ymesh, fld, meta = stagpy.field.get_meshes_fld(step, 'T',
                                                          walls=True)
    assert len(fld.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == fld.shape[0] + 1
    assert xmesh.shape[1] == ymesh.shape[1] == fld.shape[1] + 1
    assert meta.description == 'Temperature'


def test_get_meshes_vec(step):
    xmesh, ymesh, vec1, vec2 = stagpy.field.get_meshes_vec(step, 'v')
    assert len(vec1.shape) == 2
    assert xmesh.shape[0] == ymesh.shape[0] == vec1.shape[0] == vec2.shape[0]
    assert xmesh.shape[1] == ymesh.shape[1] == vec1.shape[1] == vec2.shape[1]
