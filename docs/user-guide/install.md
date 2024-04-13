Installation
============

You will need Python 3.8 or higher to use StagPy. StagPy is available on
the Python Package Index, via `pip`.

If you don't have sufficient permissions to install or update Python, you can
use [pyenv to manage Python](https://github.com/pyenv/pyenv).

Installation using `pip`
--------------------------

In most cases, installing StagPy with `pip` should be as simple as:

```sh title="shell"
python3 -m pip install stagpy
```

It might be preferable or even necessary to install StagPy in a virtual
environment to isolate it from other packages that could conflict with it:

```sh title="shell"
python3 -m venv stagpyenv
source stagpyenv/bin/activate
python3 -m pip install stagpy
```

You can then update StagPy with the following command:

```sh title="shell"
python3 -m pip install -U stagpy
```

See the [official
documentation](https://packaging.python.org/en/latest/tutorials/installing-packages/)
for more information about installing Python packages.

Some setup
----------

Run the following to create a local config file (`.stagpy.toml`):

```sh title="shell"
stagpy config --create
```

You can enable command-line auto-completion if you use either bash or zsh.

For bash:

```sh title="shell"
# adapt path as appropriate for your system
mkdir -p ~/.local/share/bash-completion/completions
cd !$
stagpy completions --bash
```

For zsh, with `fpath+=~/.zfunc` in your `.zshrc`:

```sh title="shell"
cd ~/.zfunc
stagpy completions --zsh
```

Enjoy!
