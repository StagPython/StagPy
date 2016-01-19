[![Code Health](https://landscape.io/github/mulvrova/StagPy/master/landscape.svg?style=flat-square)](https://landscape.io/github/mulvrova/StagPy/master)

# StagPy

StagPy is a command line tool to read and process StagYY output files to
produce high-quality figures.

The aim is to have different cases in one file (Cartesian, Spherical Annulus,
etc).

The code to read the binary output files has been adapted from a matlab version
initially developed by Boris Kaus.

## Installation

StagPy uses the following non-standard modules: numpy, scipy, f90nml,
matplotlib, and seaborn (the latter is optional and can be turned off with the
`core.useseaborn` option). These dependencies will be checked and needed
installation performed by `setuptools` in a `virtualenv`.

`virtualenv` is probably already installed on your computer, if this is not the
case `python2` will raise an error along the lines of `No module named
virtualenv`.  You can then install it by running `python -m pip install
virtualenv --user`. If this raise an error, see [this documentation to install
pip](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/#install-pip-setuptools-and-wheel).

The installation process is hence fairly simple:

    git clone https://github.com/mulvrova/StagPy.git
    cd StagPy
    make

A soft link named `stagpy` is created in your `~/bin` directory allows you to
launch StagPy directly by running `stagpy` in a terminal (provided that `~/bin`
is in your `PATH` environment variable).

Two files `.comp.zsh` and `.comp.sh` are created. Source them respectively in
`~/.zshrc` and `~/.bashrc` to enjoy command line completion with zsh and bash.
Run `make info` to obtain the right sourcing commands.

To check that everything work fine, go to the `data` directory of the
repository and run:

    stagpy field

Three PDF files with a plot of the temperature, pressure and
stream function fields should appear.

## Available commands

The available subcommands are the following:

- `field`: computes and/or plots scalar fields such as temperature or stream
  function;
- `rprof`: computes and/or plots radial profiles;
- `time`: computes and/or plots time series;
- `var`: displays a list of available variables;
- `config`: configuration handling.

You can run `stagpy --help` (or `stagpy -h`) to display a help message describing
those subcommands. You can also run `stagpy <subcommand> --help` to have some
help on the available options for one particular sub command.

StagPy looks for a StagYY `par` file in the current directory. It then reads
the value of the `output_file_stem` option to determine the location and name
of the StagYY output files (set to `test` if no `par` file can be found).
You can change the directory in which StagYY looks for a `par` file by two
different ways:

- you can change the default behavior in a global way by editing the config
  file (`stagpy config --edit`) and change the `core.path` variable;
- or you can change the path only for the current run with the `-p` option.

The time step option `-s` allows you to specify a range of time steps in a way
which mimic the slicing syntax: `begin:end:gap` (both ends included). If the
first step is not specified, it is set to `0`. If the final step is not
specified, all available time steps are processed. Here are some examples:

- `-s 100:350` will process every time steps between 100 and 350;
- `-s 201:206:2` will process time steps 201, 203 and 205;
- `-s 201:205:2` same as previous;
- `-s 5682:` will process every time steps from the 5682nd to the last one;
- `-s :453` will process every time steps from the 0th to the 453rd one;
- `-s ::2` will process every even time steps.

By default, the temperature, pressure and stream function fields are plotted.
You can change this with the `-o` option (e.g. `./main.py field -o ps` to plot
only the pressure and stream function fields).

