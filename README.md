[![Code Health](https://landscape.io/github/mulvrova/StagPy/master/landscape.svg?style=flat-square)](https://landscape.io/github/mulvrova/StagPy/master)

Read binary output files of STAGYY

`main.py` is the "master" file which uses definitions from the other scripts to
call the right subcommand which in turn processes StagYY data.

The available subcommands are the following:

- `field`: computes and/or plots scalar fields such as temperature or stream
  function;
- `rprof`: computes and/or plots radial profiles;
- `time`: computes and/or plots time series;
- `var`: displays a list of available variables.

StagPy uses the external modules `matplotlib` and `numpy`, please install them
if needed. To check that everything work fine, go to the `data` directory of
the repository and run `../main.py field`. Three PDF files with a plot of the
temperature, pressure and stream function fields should appear.

To make StagPy available from everywhere in your system, you can make a soft
link toward `main.py` in a directory which is in your PATH environment variable
(e.g. `ln -s $PWD/main.py ~/bin/stagpy`).

By default, StagPy looks for the binary data files in the current directory,
with a name `test_x00100`. `x` is replaced by the needed parameter name (e.g.
`t` if you want to read the temperature data file).

You can change the default behaviour by editing the `defaut_config` variable
definition in the `constants.py` module. You can also ask for a specific file
from the command line. For example, if your data file is `output/bin_x05600`,
you can access it with `./main.py -p output -n bin -s 5600 field` (see
`./main.py -h` for a complete list of options).

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
You can change this with the `-o` option (e.g. `./main.py -o ps field` to plot
only the pressure and stream function fields).

The aim is to have different cases in one file (Cartesian, Spherical Annulus,
etc).

The code to read the binary output files has been adapted from a matlab version
initially developed by Boris Kaus.
