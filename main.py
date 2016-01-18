#!/usr/bin/env python2
"""
Read and plot stagyy binary data

Author: Martina Ulvrova
Date: 2014/12/02
"""

import config

def main_func():
    """Launch appropriate subcmd"""
    args = config.parse_args()
    args.func(args)
    

if __name__ == '__main__':
    main_func()
