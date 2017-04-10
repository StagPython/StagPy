# PYTHON_ARGCOMPLETE_OK
"""The stagpy module is callable"""

from . import config


def main():
    """StagPy entry point"""
    args = config.parse_args()
    args.func(args)

main()
