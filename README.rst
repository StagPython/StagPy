.. image:: https://landscape.io/github/mulvrova/StagPy/master/landscape.svg?style=flat-square
   :target: https://landscape.io/github/mulvrova/StagPy/master
   :alt: Code Health

StagPy
======

StagPy is a Python 3 command line tool to read and process StagYY output files
to produce high-quality figures.

The aim is to have different cases in one file (Cartesian, Spherical Annulus,
etc).

The code to read the binary output files has been adapted from a matlab version
initially developed by Boris Kaus.


Installation
============

*if you want to use (and modify) the development version, see the* `For
developers`_ *section at the end of this page*

You will need Python 3.3 or higher to use StagPy. You can install StagPy with
``conda`` (you will need Python 3.5) or with ``pip``. Both process are
described hereafter.

If Python3 is not installed on your system or you don't have sufficient
permissions to update it, the simplest way to get it is to install Miniconda_
or Anaconda_ (Anaconda being Miniconda with a lot of extra modules that can be
installed in Miniconda later, this choice doesn't matter; pick Miniconda if you
want a faster and lighter installation). Then, use ``conda`` to install StagPy.

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Anaconda: https://www.continuum.io/downloads

Installation using ``conda``
----------------------------

The installation is rather simple::

    conda install -c amorison stagpy

See the `Some setup`_ subsection to enable autocompletion and create your
config file.

Installation using ``pip``
--------------------------

If you don't have ``pip`` for Python3 on your system, download the official
script <https://bootstrap.pypa.io/get-pip.py> and run it with ``python3``.

You can then install StagPy with the following command::

    python3 -m pip install --user stagpy

Make sure that the directory where ``pip`` install package entry-points
(usually ``~/.local/bin``) is in your ``PATH`` environment variable.
You can run ``python3 -m pip show stagpy`` to obtain some hint about this
location (this command will show you were the compiled sources are installed,
e.g. ``~/.local/lib/python3.5/site-packages``, from which you can deduce the
entry-point location, e.g. ``~/.local/bin``).

See the `Some setup`_ subsection to enable autocompletion and create your
config file.

Some setup
----------

Once you have installed, you can enable command-line auto-completion if you use
either bash or zsh.

Add this to your ``~/.bashrc`` file::

    eval "$(register-python-argcomplete stagpy)"

Or this to your ``~/.zshrc`` file::

    autoload bashcompinit
    bashcompinit
    eval "$(register-python-argcomplete stagpy)"


Finally, run the following once to create your config file (at
``~/.config/stagpy/``)::

    stagpy config --create

Enjoy!


Available commands
==================

The available subcommands are the following:

* ``field``: computes and/or plots scalar fields such as temperature or stream
  function;
* ``rprof``: computes and/or plots radial profiles;
* ``time``: computes and/or plots time series;
* ``plates``: plate analysis;
* ``var``: displays a list of available variables;
* ``config``: configuration handling.

You can run ``stagpy --help`` (or ``stagpy -h``) to display a help message
describing those subcommands. You can also run ``stagpy <subcommand> --help``
to have some help on the available options for one particular sub command.

StagPy looks for a StagYY ``par`` file in the current directory. It then reads
the value of the ``output_file_stem`` option to determine the location and name
of the StagYY output files (set to ``test`` if no ``par`` file can be found).
You can change the directory in which StagYY looks for a ``par`` file by two
different ways:

* you can change the default behavior in a global way by editing the config
  file (``stagpy config --edit``) and change the ``core.path`` variable;
* or you can change the path only for the current run with the ``-p`` option.

The time step option ``-s`` allows you to specify a range of time steps in a
way which mimic the slicing syntax: ``begin:end:gap`` (both ends included). If
the first step is not specified, it is set to ``0``. If the final step is not
specified, all available time steps are processed. Here are some examples:

* ``-s 100:350`` will process every time steps between 100 and 350;
* ``-s 201:206:2`` will process time steps 201, 203 and 205;
* ``-s 201:205:2`` same as previous;
* ``-s 5682:`` will process every time steps from the 5682nd to the last one;
* ``-s :453`` will process every time steps from the 0th to the 453rd one;
* ``-s ::2`` will process every even time steps.

By default, the temperature, pressure and stream function fields are plotted.
You can change this with the ``-o`` option (e.g. ``./main.py field -o ps`` to
plot only the pressure and stream function fields).


For developers
==============

If you want to contribute to development of StagPy, create an account on
GitHub_ and fork the `StagPy repository`__.

.. _GitHub: https://github.com/
.. __: https://github.com/mulvrova/StagPy

The development of StagPy is made using the Git version control system. The
first three chapters of the `Git book`__ should give you all the necessary
basic knowledge to use Git for this project.

.. __: https://git-scm.com/book/en/v2

A ``Makefile`` in the git repository allows you to install StagPy in a virtual
environment with all the necessary dependencies.  However, installation of
``numpy`` and ``scipy`` involve heavy building operations, it might be better
that you (or your system administrator) install it with a package manager such
as ``homebrew`` on Mac OS or your favorite Linux package manager (or with
``conda`` if you use it).

The installation process is then fairly simple::

    git clone https://github.com/YOUR_USER_NAME/StagPy.git
    cd StagPy
    make

A soft link named ``stagpy-git`` is created in your ``~/bin`` directory,
allowing you to launch the development version of StagPy directly by running
``stagpy-git`` in a terminal (provided that ``~/bin`` is in your ``PATH``
environment variable).

Two files ``comp.zsh`` and ``comp.sh`` are created in the ``bld`` folder.
Source them respectively in ``~/.zshrc`` and ``~/.bashrc`` to enjoy command
line completion with zsh and bash.  Run ``make info`` to obtain the right
sourcing commands.

To check that everything work fine, go to the ``data`` directory of the
repository and run::

    stagpy-git field

Three PDF files with a plot of the temperature, pressure and
stream function fields should appear.


Troubleshooting
===============

*   Matplotlib related error in MacOS

    This might be due to the matplotlib backend that is not correctly set. See
    this Stack Overflow question:
    <http://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python>

*   Installation fails with ``ImportError: No module named 'encodings'``

    This seems to be due to a bug in the venv module with some Python
    installation setups. If installing Python properly with your package
    manager doesn't solve the issue, you can try installing StagPy without any
    virtual environment by using ``make novirtualenv``.
