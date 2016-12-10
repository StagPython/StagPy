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

The version installed in the virtual environment points directly towards the
source files. It means that you don't need to run ``make`` again for your
changes to the source files to be taken into account.

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

Two PDF files with a plot of the temperature and vertical velocity fields
should appear.

