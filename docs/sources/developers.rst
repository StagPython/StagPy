Contributing
============

If you want to contribute to development of StagPy, create an account on
GitHub_ and fork the `StagPy repository`__.

.. _GitHub: https://github.com/
.. __: https://github.com/StagPython/StagPy

The development of StagPy is made using the Git version control system. The
first three chapters of the `Git book`__ should give you all the necessary
basic knowledge to use Git for this project.

.. __: https://git-scm.com/book/en/v2

To get a local copy of your fork of StagPy, clone it (you can use `the SSH
protocol`__ if you prefer)::

    % git clone https://github.com/YOUR_USER_NAME/StagPy.git
    % cd StagPy

.. __: https://help.github.com/articles/connecting-to-github-with-ssh/

Testing
-------

StagPy uses pytest_ and tox_ for code testing. Make sure they are installed
and up to date on your system::

    % python3 -m pip install -U --user pytest tox

.. _pytest: https://docs.pytest.org
.. _tox: https://tox.readthedocs.io

Launching ``tox`` in the root of the repository will automatically run the
tests in a virtual environment. Before submitting modifications to the code,
please make sure they pass the tests by running ``tox``.

Documentation
-------------

The StagPy documentation is built with Sphinx_. To build it locally, install
and update the needed packages::

    % python3 -m pip install -U --user sphinx sphinx-rtd-theme

.. _Sphinx: http://www.sphinx-doc.org

Then, in the ``docs`` directory, run::

    % make html

Open the produced file ``_build/html/index.html`` in your navigator to browse
your local version of the documentation.

Installation
------------

You can install the development version in two ways:

1. in a virtual environment, allowing to have the development version alongside
   the stable one on your system;
2. as a regular package, allowing you to import the development version of
   StagPy even from outside the virtual environment.

The second option should only be used if necessary for your purpose.

Installation in a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Makefile`` in the git repository allows you to install StagPy in a virtual
environment with all the necessary dependencies.

The version installed in the virtual environment points directly towards the
source files. It means that you don't need to run ``make`` again for your
changes to the source files to be taken into account.

A soft link named ``stagpy-git`` is created in your ``~/bin`` directory,
allowing you to launch the development version of StagPy directly by running
``stagpy-git`` in a terminal (provided that ``~/bin`` is in your ``PATH``
environment variable).

See :ref:`somesetup` to enable command line completion (replacing the ``stagpy``
command with ``stagpy-git``).

To check that everything works fine, go to any subdirectory of the ``Examples``
directory of the repository and run::

    % stagpy-git field

This should create a PDF file showing a plot of the temperature field with
streamlines.

Installation as a regular package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have installed the stable version of StagPy, uninstall it::

    % python3 -m pip uninstall stagpy

You can use the following command to install StagPy as a regular package::

    % python3 -m pip install -U --user -e .

You *don't* need to run this command everytime you modify the source files.
If you want to uninstall the development version, you can simply run::

    % python3 -m pip uninstall stagpy

See :ref:`somesetup` to enable command line completion.
