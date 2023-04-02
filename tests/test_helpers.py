import stagpy
import stagpy._helpers


def test_out_name_conf() -> None:
    oname = "something_fancy"
    stagpy.conf.core.outname = oname
    stem = "teapot"
    assert stagpy._helpers.out_name(stem) == oname + "_" + stem
    del stagpy.conf.core.outname


def test_out_name_number() -> None:
    assert stagpy._helpers.out_name("T", 123) == "stagpy_T00123"


def test_baredoc() -> None:
    """
       Badly formatted docstring .. .

    With some content.

    """
    expected = "Badly formatted docstring"
    assert stagpy._helpers.baredoc(test_baredoc) == expected
