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

These options are used by the ``config`` subcommand.

.. option:: --create

   Create a new config file from scratch. This is useful if your configuration
   file is broken or you want to reset the modifications you made to your
   configuration.

.. option:: --update

   Add missing entries to your config file (or create a new one if necessary).

.. option:: --edit

   Open your config file in ``vim``.

