#!/usr/bin/env python2
"""
Read and plot stagyy binary data

Author: Martina Ulvrova
Date: 2014/12/02
"""

import argparse

import commands
import constants
import misc

# top level parser
parser = argparse.ArgumentParser(
    description='read and process StagYY binary data')

# options common to every sub-commands
parser.add_argument('-g', '--geometry', choices=['annulus'],
                    help='geometry of the domain')
parser.add_argument('-p', '--path',
                    help='StagYY output directory')
parser.add_argument('-n', '--name',
                    help='StagYY generic output file name')
parser.add_argument('-s', '--timestep', help='timestep')
parser.add_argument('-o', '--plot', nargs='?', const='',
                    help='specify which variables to plot, use --var \
                    option for a list of available variables')
parser.add_argument('--dsa', type=float,
                    help='thickness of the sticky air')
parser.add_argument('--shrinkcb', type=float,
                    help='color bar shrink')
parser.add_argument('--pdf', action='store_true',
                    help='produces non-rasterized, high quality \
                    pdf (slow!)')

parser.set_defaults(**constants.default_config)

subparsers = parser.add_subparsers()

# parser for the "field" command
parser_fd = subparsers.add_parser('field')
parser_fd.set_defaults(func=commands.field)

# parser for the "rprof" command
parser_rp = subparsers.add_parser('rprof')
parser_rp.set_defaults(func=commands.rprof)

# parser for the "time" command
parser_tm = subparsers.add_parser('time')
parser_tm.set_defaults(func=commands.time)

# parser for the "var" command
parser_var = subparsers.add_parser('var')
parser_var.set_defaults(func=commands.var)

args = parser.parse_args()
args = misc.parse_timesteps(args)
args.func(args)
