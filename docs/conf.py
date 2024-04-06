# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from dataclasses import fields
from pathlib import Path
from textwrap import dedent

from pkg_resources import get_distribution

sys.path.insert(0, os.path.abspath(".."))

import stagpy

html_theme = "sphinx_rtd_theme"

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "7.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

root_doc = "index"

autodoc_member_order = "bysource"
autoclass_content = "class"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- Project information -----------------------------------------------------
project = "StagPy"
copyright = "2015 - 2023, Adrien Morison, Martina Ulvrova, Stéphane Labrosse"
author = "Adrien Morison, Martina Ulvrova, Stéphane Labrosse"

# The full version, including alpha/beta/rc tags.
release = get_distribution("stagpy").version
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output ----------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "StagPydoc"


# -- Autogenerated configuration options list -----------------------------

dfile = Path(".") / "sources" / "config_opts.rst"
with dfile.open("w") as fid:
    fid.write(
        dedent(
            """\
        ..
           This doc is automatically generated in conf.py.
           Editing it will have no effect.
           Modify conf.py instead.

        List of configuration options
        =============================

        These tables list configuration options.
        """
        )
    )
    for sec_fld in fields(stagpy.conf):
        sec_name = sec_fld.name
        fid.write(
            dedent(
                """
            .. list-table:: {}
               :header-rows: 1

               * - Name
                 - Description
                 - CLI, config file?
            """.format(sec_name)
            )
        )
        section = getattr(stagpy.conf, sec_name)
        for fld in fields(section):
            opt = fld.name
            entry = section.meta_(opt).entry
            if entry.in_cli and entry.in_file:
                c_f = "both"
            elif entry.in_cli:
                c_f = "CLI"
            else:
                c_f = "config file"
            fid.write("   * - {}\n".format(opt))
            fid.write("     - {}\n".format(entry.doc))
            fid.write("     - {}\n".format(c_f))
