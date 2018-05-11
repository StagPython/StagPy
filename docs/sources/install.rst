Installation
============

You will need Python 3.5 or higher to use StagPy. StagPy is available on
the Python Package Index, via ``pip``.

If Python3 is not installed on your system or you don't have sufficient
permissions to update it, the simplest way to get it is to install Miniconda_
or Anaconda_ (Anaconda being Miniconda with a lot of extra modules that can be
installed in Miniconda later, this choice doesn't matter; pick Miniconda if you
want a faster and lighter installation). Then, use ``conda`` to install the
following dependencies::

    % conda install numpy scipy pandas h5py matplotlib seaborn setuptools_scm

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Anaconda: https://www.continuum.io/downloads

Installation using ``pip``
--------------------------

If you don't have ``pip`` for Python3 on your system, use the ``ensurepip``
module to install it (it is bootstrapped within the Python interpreter)::

    % python3 -m ensurepip --user

In case this doesn't work, download the official script
<https://bootstrap.pypa.io/get-pip.py> and run it with ``python3``.

You can then install and/or update StagPy with the following command::

    % python3 -m pip install -U --user stagpy

Make sure that the directory where ``pip`` install package entry-points
(usually ``~/.local/bin``) is in your ``PATH`` environment variable.
You can run ``python3 -m pip show stagpy`` to obtain some hint about this
location (this command will show you were the compiled sources are installed,
e.g. ``~/.local/lib/python3.5/site-packages``, from which you can deduce the
entry-point location, e.g. ``~/.local/bin``).

See the `Some setup`_ subsection to enable autocompletion and create your
config file.

.. _somesetup:

Some setup
----------

Run the following once to create your config file (in ``~/.config/stagpy/``)::

    % stagpy config --create

You can enable command-line auto-completion if you use either bash or zsh.

Add this to your ``~/.bashrc`` file::

    source ~/.config/stagpy/bash/stagpy.sh

Or this to your ``~/.zshrc`` file::

    source ~/.config/stagpy/zsh/_stagpy.sh

Enjoy!

