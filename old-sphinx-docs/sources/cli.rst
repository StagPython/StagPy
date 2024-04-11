Command line interface
======================

The command line interface is organized in subcommands. Three subcommands
(``var``, ``version`` and ``config``) deals with StagPy related stuff, while
the others (``field``, ``rprof``, ``time`` and ``plates``) handles the
processing of StagYY output data. The latter set shares some generic options
which are described in the following subsection.

Generic options
---------------

These options are shared by the ``field``, ``rprof``, ``time`` and ``plates``
subcommands.

.. option:: -p <path>, --path <path>

   Path towards the StagYY run directory (i.e. the directory containing the par
   file of your simulation). The path can be absolute or relative to your
   working directory. If not specified, the working directory is assumed to be
   the StagYY run directory.

.. option:: --outname <outname>

   Prefix used to name the files produced by StagPy. "stagpy" is the default
   value.

.. option:: -t <step-slice>, --timesteps <step-slice>

   Range of timesteps that should be processed. Defaults to the last available
   snapshot.

.. option:: -s <snap-slice>, --snapshots <snap-slice>

   Range of snapshots that should be processed. Defaults to the last available
   snapshot.

.. option:: -xkcd, +xkcd

    Toggle xkcd plot style. Defaults to disabled.

.. option:: -pdf, +pdf

    Toggle rasterization of produced figures. Defaults to enabled rasterization
    (``-pdf``).

Configuration options
---------------------

These options are used by the ``config`` subcommand. If none of these is used,
a list of the available configuration options along with a short help message
is displayed.

.. option:: --create

   Create a new config file from scratch.

