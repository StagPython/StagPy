import pytest
import stagpy
import stagpy.error
from stagpy.config import _CONF_DEF as dflt

def test_get_subconfig():
    conf = stagpy.config.StagpyConfiguration(None)
    for sub in dflt:
        assert getattr(conf, sub) is conf[sub]

def test_get_opt():
    conf = stagpy.config.StagpyConfiguration(None)
    for sub, opt in conf.options():
        assert getattr(conf[sub], opt) is conf[sub][opt]

def test_get_invalid_subconfig():
    conf = stagpy.config.StagpyConfiguration(None)
    invalid = 'invalidsubdummy'
    with pytest.raises(stagpy.error.ConfigSectionError) as err:
        _ = conf[invalid]
    assert err.value.section == invalid

def test_get_invalid_opt():
    conf = stagpy.config.StagpyConfiguration(None)
    invalid = 'invalidoptdummy'
    with pytest.raises(stagpy.error.ConfigOptionError) as err:
        _ = conf.core[invalid]
    assert err.value.option == invalid

def test_reset_subconfig():
    conf = stagpy.config.StagpyConfiguration(None)
    default = conf.core.path
    conf.core.path = default + 'some/path'
    del conf.core
    assert conf.core.path == default

def test_reset_subconfig_item():
    conf = stagpy.config.StagpyConfiguration(None)
    default = conf.core.path
    conf.core.path = default + 'some/path'
    del conf['core']
    assert conf.core.path == default

def test_reset_opt():
    conf = stagpy.config.StagpyConfiguration(None)
    default = conf.core.path
    conf.core.path = default + 'some/path'
    del conf.core.path
    assert conf.core.path == default

def test_reset_opt_item():
    conf = stagpy.config.StagpyConfiguration(None)
    default = conf.core.path
    conf.core.path = default + 'some/path'
    del conf.core['path']
    assert conf.core.path == default

def test_config_iter_subs():
    conf = stagpy.config.StagpyConfiguration(None)
    raw_iter = set(iter(conf))
    subs_iter = set(conf.subs())
    subs_expected = set(dflt.keys())
    assert raw_iter == subs_iter == subs_expected

def test_config_iter_options():
    conf = stagpy.config.StagpyConfiguration(None)
    options_iter = set(conf.options())
    options_expected = set((sub, opt) for sub in dflt for opt in dflt[sub])
    assert options_iter == options_expected

def test_config_iter_default_val():
    conf = stagpy.config.StagpyConfiguration(None)
    vals_iter = set(conf.opt_vals())
    vals_dflts = set((s, o, m.default) for s, o, m in conf.defaults())
    assert vals_iter == vals_dflts

def test_config_iter_subconfig():
    conf = stagpy.config.StagpyConfiguration(None)
    raw_iter = set(iter(conf.core))
    opts_iter = set(conf.core.options())
    opts_expected = set(dflt['core'].keys())
    assert raw_iter == opts_iter == opts_expected

def test_config_iter_subconfig_default_val():
    conf = stagpy.config.StagpyConfiguration(None)
    vals_iter = set(conf.core.opt_vals())
    vals_dflts = set((o, m.default) for o, m in conf.core.defaults())
    assert vals_iter == vals_dflts
