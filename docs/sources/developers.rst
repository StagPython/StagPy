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

    git clone https://github.com/YOUR_USER_NAME/StagPy.git
    cd StagPy

.. __: https://help.github.com/articles/connecting-to-github-with-ssh/

You can then install the development version in two ways:

1. in a virtual environment, allowing to have the development version alongside
   the stable one on your system;
2. as a regular package, allowing you to import the development version of
   StagPy even from outside the virtual environment.

The second option should only be used if necessary for your purpose.

Installation in a virtual environment
-------------------------------------

A ``Makefile`` in the git repository allows you to install StagPy in a virtual
environment with all the necessary dependencies.

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

Installation as a regular package
---------------------------------

If you have installed the stable version of StagPy, uninstall it::

    python3 -m pip uninstall stagpy

You can use the following command to install StagPy as a regular package::

    python3 setup.py develop --user

You *don't* need to run this command everytime you modify the source files.
If you want to uninstall the development version, you can run::

    python3 setup.py develop --user --uninstall

Add the following to your ``.zshrc``::

    autoload bashcompinit
    bashcompinit
    eval "$(register-python-argcomplete stagpy)"

or only this line to your ``.bashrc``::

    eval "$(register-python-argcomplete stagpy)"

to enjoy command line completion with zsh and bash.
