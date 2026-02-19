Installation
============

StagPy is available on the [Python Package
Index](https://pypi.org/project/stagpy/).

You can use [uv to manage Python environments](https://docs.astral.sh/uv/).

Installation as a CLI tool
--------------------------

If you are interested in using the `stagpy` command line interface,
you can install it as a tool with uv:

```sh title="shell"
uv tool install stagpy
```

This installs `stagpy` in its own environment, isolated from other packages to
avoid conflicts.

You can then update StagPy with the following command:

```sh title="shell"
uv tool upgrade stagpy
```

[More information about uv tools](https://docs.astral.sh/uv/concepts/tools/).

Installation in a uv managed environment
----------------------------------------

A convenient way to use StagPy in script is by leveraging uv environments.

With the following (setting versions as desired for your project):

```toml title="pyproject.toml"
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "stagpy~=0.22.0",
]
```

You can then run a script using `stagpy`, for example:

```py title="my_script.py"
import stagpy

print(stagpy.__version__)
```

with the following command

```sh title="shell"
uv run my_script.py
```

You can run any arbitrary command via uv, including `stagpy`
itself:

```sh title="shell"
uv run -- stagpy version
```


Shell completions
-----------------

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
