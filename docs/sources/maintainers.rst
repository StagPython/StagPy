Releasing process
=================

This section is intended for maintainers of the project. It describes how new
versions of StagPy are released on PyPI.

Version numbers are tracked with git tags thanks to ``setuptools_scm``. Marking
a new version merely consists in tagging the ``HEAD`` of the ``master`` branch.
Please make sure to always provide a patch version number (i.e. use a version
number with *three* levels such as ``1.0.0`` instead of ``1.0``).

::

    % git tag -a vX.Y.Z
    % git push --follow-tags

Releasing on PyPI is a two steps process:

1. construct the wheel and source package;
2. upload those on PyPI using ``twine`` (you will need a PyPI password).

::

    % python3 -m pip install -U build twine
    % python3 -m build
    % python3 -m twine upload dist/*
    % rm -rf dist/

