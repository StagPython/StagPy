#!/usr/bin/env python2
"""
Read and plot stagyy binary data

Author: Martina Ulvrova
Date: 2014/12/02
"""

import commands
import config
import misc

def main_func():
    """Launch appropriate subcmd"""
    parser = config.create_parser()
    args = parser.parse_args()
    if not args.func is commands.var_cmd:
        args = misc.parse_timesteps(args)
        args = misc.plot_backend(args)
    args.func(args)
    

if __name__ == '__main__':
    main_func()
