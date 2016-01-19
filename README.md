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
installation performed by `setuptools`.

The installation process is hence fairly simple:

    git clone https://github.com/mulvrova/StagPy.git
    cd StagPy
    python2 setup.py install --user

You might want to drop the `--user` option if you install StagPy in a
virtualenv or as root.

`setuptools` should create a script named `stagpy` that allows you to run
StagPy directly by running `stagpy` in a terminal. The installation location of
this script is shown in the output of the last command, which should contain
the following line (right before the dependencies related outputs):

    Installing stagpy script to /home/username/.local/bin

This directory will have to be in your `PATH` environment variable for the
`stagpy` command to be available.

To create the global config file used by `stagpy`, run:

    stagpy config --create

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

