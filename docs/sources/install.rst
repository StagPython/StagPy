Installation
============

You will need Python 3.8 or higher to use StagPy. StagPy is available on
the Python Package Index, via ``pip``.

If you don't have sufficient permissions to install or update Python, you can
use `pyenv to manage Python <https://github.com/pyenv/pyenv>`_.

Installation using ``pip``
--------------------------

In most cases, installing StagPy with ``pip`` should be as simple as::

    % python3 -m pip install stagpy

It might be preferable or even necessary to install StagPy in a virtual
environment to isolate it from other packages that could conflict with it::

    % python3 -m venv stagpyenv
    % source stagpyenv/bin/activate
    % python3 -m pip install stagpy

You can then update StagPy with the following command::

    % python3 -m pip install -U stagpy

See the
`official documentation <https://packaging.python.org/en/latest/tutorials/installing-packages/>`_
for more information about installing Python packages.

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
