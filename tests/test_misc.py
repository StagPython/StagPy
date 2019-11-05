import stagpy
import stagpy.misc


def test_out_name_conf():
    oname = 'something_fancy'
    stagpy.conf.core.outname = oname
    stem = 'teapot'
    assert stagpy.misc.out_name(stem) == oname + '_' + stem
    del stagpy.conf.core.outname


def test_out_name_number():
    assert stagpy.misc.out_name('T', 123) == 'stagpy_T00123'


def test_baredoc():
    """
       Badly formatted docstring .. .

    With some content.

    """
    expected = 'Badly formatted docstring'
    assert stagpy.misc.baredoc(test_baredoc) == expected


def test_list_of_vars():
    expected = [[['a', 'b'], ['c', 'd', 'e']], [['f', 'g'], ['h']]]
    assert stagpy.misc.list_of_vars('a,b..c,d,,,e-f,g.h-,..,-') == expected


def test_set_of_vars():
    expected = set(iter('abcdefgh'))
    lovs = [[['a', 'b'], ['c', 'd', 'e']], [['f', 'g'], ['h']]]
    assert stagpy.misc.set_of_vars(lovs) == expected
