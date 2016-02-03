"""
Read and plot stagyy binary data

Author: Martina Ulvrova
Date: 2014/12/02
"""

from . import config


def main():
    """stagpy entry point"""
    args = config.parse_args()
    args.func(args)
