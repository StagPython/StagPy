.. image:: https://landscape.io/github/StagPython/StagPy/master/landscape.svg?style=flat-square
   :target: https://landscape.io/github/StagPython/StagPy/master
   :alt: Code Health

.. image:: https://readthedocs.org/projects/stagpy/badge/?version=latest
   :target: http://stagpy.readthedocs.org/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://badge.fury.io/py/stagpy.svg
   :target: https://badge.fury.io/py/stagpy
   :alt: PyPI Version


StagPy
======

StagPy is a command line tool to process the output files of your StagYY
simulations and produce high-quality figures.

This command line tool is built around a generic interface that allows you to
access StagYY output data directly in a Python script.

You can install StagPy with ``pip`` from the `Python Package Index`__.

See `the complete documentation`__ for more information on the installation and
explanations on how to use StagPy.

.. __: https://pypi.python.org/pypi/stagpy
.. __: http://stagpy.readthedocs.org/en/latest/


Troubleshooting
===============

*   Matplotlib related error in MacOS

    This might be due to the matplotlib backend that is not correctly set. See
    this Stack Overflow question:
    <http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>
