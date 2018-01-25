from pkg_resources import safe_version
import stagpy

def test_version_format():
    assert stagpy.__version__ == safe_version(stagpy.__version__)
