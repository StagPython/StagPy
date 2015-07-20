Read binary output files of STAGYY

`main.py` is the "master" file which uses definitions from the other scripts to
read and process the data files.

To make StagPy available from everywhere in your system, you can make a soft
link toward `main.py` in a directory which is in your PATH environment variable
(e.g. `ln -s $PWD/main.py ~/bin/stagpy`).

By default, StagPy looks for the binary data files in the current directory,
with a name `test_x00100`. `x` is replaced by the needed parameter name (e.g.
`t` if you want to read the temperature data file).

You can change the default behaviour by editing the `defaut_config` variable
definition in the `constants.py` module. You can also ask for a specific file
from the command line. For example, if your data file is `output/bin_x05600`,
you can access it with `./main.py -p output -n bin -s 5600` (see `./main.py -h`
for a complete list of options).

By default, the temperature, pressure and stream function fields are plotted.
You can change this with the `-o` option (e.g. `./main.py -o ps` to plot only
the pressure and stream function fields). See `./main.py --var` for a complete
list of available variables.

The aim is to have different cases in one file (Cartesian, Spherical Annulus, etc)

The code to read the binary output files has been adapted from a matlab version initially developed by Boris Kaus.

