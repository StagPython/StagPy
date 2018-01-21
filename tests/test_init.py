from pkg_resources import safe_version
import stagpy
import helpers

def test_init_config_None():
    stagpy.conf.core.path = 'some_path'
    helpers.reset_config()
    assert stagpy.conf.core.path == './'

def test_version_format():
    assert stagpy.__version__ == safe_version(stagpy.__version__)
