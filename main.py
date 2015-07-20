#!/usr/bin/env python2
"""
  Read and plot stagyy binary data
  Author: Martina Ulvrova
  Date: 2014/12/02
"""

from __future__ import print_function
import argparse
import sys

from stag import StagyyData
import constants

parser = argparse.ArgumentParser(
    description='read and process StagYY binary data')
parser.add_argument('-g', '--geometry', choices=['annulus'],
                    help='geometry of the domain')
parser.add_argument('-p', '--path',
                    help='StagYY output directory')
parser.add_argument('-n', '--name',
                    help='StagYY generic output file name')
parser.add_argument('-s', '--timestep', type=int,
                    help='timestep')
parser.add_argument('-o', '--plot', nargs='?', const='',
                    help='specify which variables to plot, use --var \
                    option for a list of available variables')
parser.add_argument('--dsa', type=float,
                    help='thickness of the sticky air')
parser.add_argument('--shrinkcb', type=float,
                    help='color bar shrink')
parser.add_argument('--var', action='store_true',
                    help='display a list of available variables')

parser.set_defaults(**constants.default_config)
args = parser.parse_args()

if args.var:
    print(*('{}: {}'.format(k, v.name) for k, v in constants.varlist.items()),
          sep='\n')
    sys.exit()

for var in set(args.plot).intersection(constants.varlist):
    stgdat = StagyyData(args, constants.varlist[var].par)
    stgdat.plot_scalar(var)
