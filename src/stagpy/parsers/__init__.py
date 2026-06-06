# ruff: noqa: F401

"""Parsers of StagYY output files.

Note:
    These functions are low level utilities. You should not use these unless
    you know what you are doing. To access StagYY output data, use an instance
    of [`StagyyData`][stagpy.stagyydata.StagyyData].
"""

from . import bin, h5, txt
