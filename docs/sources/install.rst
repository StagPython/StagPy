Installation
============

You will need Python 3.4 or higher to use StagPy. You can install StagPy with
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

    python3 -m pip install -U --user stagpy

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


