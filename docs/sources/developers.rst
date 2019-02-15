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

StagPy uses tox_ for code testing.  Make sure it is installed and up to date on
your system::

    % python3 -m pip install -U --user tox

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

Installation in a virtual environment
-------------------------------------

A ``Makefile`` in the git repository allows you to install the development
version of StagPy in a virtual environment with all the necessary dependencies.
Simply run ``make`` to do so.

The version installed in the virtual environment points directly towards the
source files. It means that you don't need to run ``make`` again for your
changes to the source files to be taken into account.

To activate the virtual environment, source the relevant script::

    % source .venv_dev/bin/activate

Once this is done, launching the ``stagpy`` command and importing StagPy in a
Python script (``import stagpy``) will use the development version of StagPy.
Launch the ``deactivate`` command to get out of the virtual environment.

As a convenience, a soft link named ``stagpy-git`` is created in your ``~/bin``
directory, allowing you to launch the development version CLI tool without
activating the virtual environment by running ``stagpy-git`` in a terminal
(provided that ``~/bin`` is in your ``PATH`` environment variable).

See :ref:`somesetup` to enable command line completion (replacing the ``stagpy``
command with ``stagpy-git``).

To check that everything works fine, go to any subdirectory of the ``Examples``
directory of the repository and run::

    % stagpy-git field

This should create a PDF file showing a plot of the temperature field with
streamlines.
