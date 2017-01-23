Releasing process
=================

This section is intended for maintainers of the project. It describes how new
versions of StagPy are released on PyPI and the Anaconda Cloud.

Version numbers are tracked with git tags thanks to ``setuptools_scm``. Marking
a new version merely consists in tagging the ``HEAD`` of the ``master`` branch.
Please make sure to always provide a patch version number (i.e. use a version
number with *three* levels such as ``1.0.0`` instead of ``1.0``).

::

    % git tag -a vX.Y.Z
    % git push --tags

Releasing on PyPI is a two steps process:

1. construct the wheel and source package using ``setuptools`` (and ``wheel``);
2. upload those on PyPI using ``twine`` (you will need a PyPI password).

::

    % python3 setup.py sdist bdist_wheel
    % python3 -m twine upload dist/*
    % rm -rf build/ dist/ stagpy.egg-info

The release on the Anaconda cloud will use the package uploaded on PyPI to
build the conda package. Make sure you have the ``conda-build`` and
``anaconda-client`` packages installed in your conda environment. If necessary,
add the channel ``conda config --add channels amorison``.

::

    % mkdir conda-bld; cd conda-bld
    % conda skeleton pypi stagpy
    % conda build --no-anaconda-upload --no-test stagpy
    % conda convert --platform all -o all /path/to/stagpy-*.tar.bz2
    % anaconda login
    % anaconda upload all/*/*
    % anaconda logout
    % cd ..; rm -rf conda-bld

The ``/path/to/stagpy-*.tar.bz2`` is given in the output of the ``conda build``
command.
