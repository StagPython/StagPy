from pathlib import Path

import pytest

import stagpy.error
from stagpy.parfile import StagyyPar


def test_no_par_file() -> None:
    invalid = Path("dummyinvalidpar")
    with pytest.raises(stagpy.error.NoParFileError) as err:
        StagyyPar.from_main_par(invalid)
    assert err.value.parfile == invalid
