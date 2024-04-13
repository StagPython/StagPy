"""StagPy is a tool to postprocess StagYY output files.

StagPy is both a CLI tool and a powerful Python library. See the
documentation at https://stagpython.github.io/StagPy/

When using the CLI interface, warnings are ignored and only a short form of
encountered StagpyErrors is printed. Set the environment variable STAGPY_DEBUG
to issue warnings normally and raise StagpyError.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
