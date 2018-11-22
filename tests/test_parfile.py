import pathlib
import pytest
import stagpy.error
import stagpy.parfile


@pytest.fixture
def par_nml(example_dir):
    return stagpy.parfile.readpar(example_dir / 'par', example_dir)


def test_section_present(par_nml):
    for section in stagpy.parfile.PAR_DEFAULT.keys():
        assert section in par_nml


def test_section_case_insensitive(par_nml):
    return par_nml['SwitcHes'] is par_nml['switches']


def test_option_case_insensitive(par_nml):
    return par_nml['refstate']['Ra0'] is par_nml['refstate']['ra0']


def test_no_par_file():
    invalid = pathlib.Path('dummyinvalidpar')
    with pytest.raises(stagpy.error.NoParFileError) as err:
        stagpy.parfile.readpar(invalid, invalid)
    assert err.value.parfile == invalid
