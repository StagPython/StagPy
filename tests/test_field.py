import stagpy.field
import stagpy.phyvars


def test_valid_field_var():
    for var in stagpy.phyvars.FIELD:
        assert stagpy.field.valid_field_var(var)
    for var in stagpy.phyvars.FIELD_EXTRA:
        assert stagpy.field.valid_field_var(var)


def test_valid_field_var_invalid():
    assert not stagpy.field.valid_field_var('dummyfieldvar')


def test_get_meshes_fld(step):
    xmesh, ymesh, fld = stagpy.field.get_meshes_fld(step, 'T')
    assert len(fld.shape) == 2
    assert xmesh.shape == ymesh.shape == fld.shape


def test_get_meshes_vec(step):
    xmesh, ymesh, vec1, vec2 = stagpy.field.get_meshes_vec(step, 'v')
    assert len(vec1.shape) == 2
    assert xmesh.shape == ymesh.shape == vec1.shape == vec2.shape
