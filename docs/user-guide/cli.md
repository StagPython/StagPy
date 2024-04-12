Command line interface
======================

The command line interface is organized in subcommands. Three subcommands
(`var`, `version` and `config`) deals with StagPy related stuff, while
the others (`field`, `rprof`, `time` and `plates`) handles the
processing of StagYY output data. The latter set shares some generic options
which are described in the following subsection.

Generic options
---------------

These options are shared by the `field`, `rprof`, `time` and `plates`
subcommands.

- `-p <path>, --path <path>`: path towards the StagYY run directory (i.e. the
  directory containing the par file of your simulation). The path can be
  absolute or relative to your working directory. If not specified, the working
  directory is assumed to be the StagYY run directory.

- `--outname <outname>`: prefix used to name the files produced by StagPy.
  `"stagpy"` is the default value.

- `-t <step-slice>, --timesteps <step-slice>`: range of timesteps that should
  be processed. Defaults to the last available snapshot.

- `-s <snap-slice>, --snapshots <snap-slice>`: range of snapshots that should
  be processed. Defaults to the last available snapshot.

- `--xkcd`: enable xkcd plot style.

- `-raster, +raster`: toggle rasterization of produced figures. Defaults to
  enabled rasterization.

Configuration options
---------------------

These options are used by the `config` subcommand. If none of these is used,
a list of the available configuration options along with a short help message
is displayed.

- `--create`: create a new config file `.stagpy.toml` from scratch.

