.. image:: https://landscape.io/github/StagPython/StagPy/master/landscape.svg?style=flat-square
   :target: https://landscape.io/github/StagPython/StagPy/master
   :alt: Code Health

.. image:: https://readthedocs.org/projects/stagpy/badge/?version=latest
   :target: http://stagpy.readthedocs.org/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://anaconda.org/amorison/stagpy/badges/version.svg
   :target: https://anaconda.org/amorison/stagpy
   :alt: Anaconda Cloud

.. image:: https://badge.fury.io/py/stagpy.svg
   :target: https://badge.fury.io/py/stagpy
   :alt: PyPI Version


StagPy
======

StagPy is a Python 3 command line tool to read and process StagYY output files
to produce high-quality figures.

The aim is to have different cases in one file (Cartesian, Spherical Annulus,
etc).

The code to read the binary output files has been adapted from a matlab version
initially developed by Boris Kaus.

You can install StagPy with ``conda`` from the `Anaconda Cloud`__, or with
``pip`` from the `Python Package Index`__.

See `the complete documentation`__ for more information on the installation and
explanations on how to use StagPy.

.. __: https://anaconda.org/amorison/stagpy
.. __: https://pypi.python.org/pypi/stagpy
.. __: http://stagpy.readthedocs.org/en/latest/


Troubleshooting
===============

*   Matplotlib related error in MacOS

    This might be due to the matplotlib backend that is not correctly set. See
    this Stack Overflow question:
    <http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>
