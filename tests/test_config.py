import pytest
import stagpy.error
from stagpy.config import _CONF_DEF as dflt

def test_get_subconfig(cleanconf):
    for sub in dflt:
        assert getattr(cleanconf, sub) is cleanconf[sub]

def test_get_opt(cleanconf):
    for sub, opt in cleanconf.options():
        assert getattr(cleanconf[sub], opt) is cleanconf[sub][opt]

def test_get_invalid_subconfig(cleanconf):
    invalid = 'invalidsubdummy'
    with pytest.raises(stagpy.error.ConfigSectionError) as err:
        _ = cleanconf[invalid]
    assert err.value.section == invalid

def test_get_invalid_opt(cleanconf):
    invalid = 'invalidoptdummy'
    with pytest.raises(stagpy.error.ConfigOptionError) as err:
        _ = cleanconf.core[invalid]
    assert err.value.option == invalid

def test_reset_all(cleanconf):
    default = cleanconf.core.path
    cleanconf.core.path = default + 'some/path'
    cleanconf.reset()
    assert cleanconf.core.path == default

def test_reset_subconfig(cleanconf):
    default = cleanconf.core.path
    cleanconf.core.path = default + 'some/path'
    del cleanconf.core
    assert cleanconf.core.path == default

def test_reset_subconfig_item(cleanconf):
    default = cleanconf.core.path
    cleanconf.core.path = default + 'some/path'
    del cleanconf['core']
    assert cleanconf.core.path == default

def test_reset_opt(cleanconf):
    default = cleanconf.core.path
    cleanconf.core.path = default + 'some/path'
    del cleanconf.core.path
    assert cleanconf.core.path == default

def test_reset_opt_item(cleanconf):
    default = cleanconf.core.path
    cleanconf.core.path = default + 'some/path'
    del cleanconf.core['path']
    assert cleanconf.core.path == default

def test_config_iter_subs(cleanconf):
    raw_iter = set(iter(cleanconf))
    subs_iter = set(cleanconf.subs())
    subs_expected = set(dflt.keys())
    assert raw_iter == subs_iter == subs_expected

def test_config_iter_options(cleanconf):
    options_iter = set(cleanconf.options())
    options_expected = set((sub, opt) for sub in dflt for opt in dflt[sub])
    assert options_iter == options_expected

def test_config_iter_default_val(cleanconf):
    vals_iter = set(cleanconf.opt_vals())
    vals_dflts = set((s, o, m.default) for s, o, m in cleanconf.defaults())
    assert vals_iter == vals_dflts

def test_config_iter_subconfig(cleanconf):
    raw_iter = set(iter(cleanconf.core))
    opts_iter = set(cleanconf.core.options())
    opts_expected = set(dflt['core'].keys())
    assert raw_iter == opts_iter == opts_expected

def test_config_iter_subconfig_default_val(cleanconf):
    vals_iter = set(cleanconf.core.opt_vals())
    vals_dflts = set((o, m.default) for o, m in cleanconf.core.defaults())
    assert vals_iter == vals_dflts
