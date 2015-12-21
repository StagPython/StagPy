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

class Toggle(argparse.Action):
    def __call__(self, parser, namespace, values, option):
        setattr(namespace, self.dest, bool('-+'.index(option[0])))

def main_func():
    """deals with command line arguments"""
    # top level parser
    main_parser = argparse.ArgumentParser(
        description='read and process StagYY binary data')
    subparsers = main_parser.add_subparsers()

    # options common to every sub-commands
    parent_parser = argparse.ArgumentParser(add_help=False, prefix_chars='-+')
    parent_parser.add_argument('-g', '--geometry', choices=['annulus'],
                        help='geometry of the domain')
    parent_parser.add_argument('-p', '--path',
                        help='StagYY output directory')
    parent_parser.add_argument('-n', '--name',
                        help='StagYY generic output file name')
    parent_parser.add_argument('-s', '--timestep', help='timestep')
    parent_parser.add_argument('-o', '--plot', nargs='?', const='',
                        help='specify which variables to plot, use var \
                        subcommand for a list of available variables')
    parent_parser.add_argument('--dsa', type=float,
                        help='thickness of the sticky air')
    parent_parser.add_argument('--linewidth', type=int,
                        help='line width')
    parent_parser.add_argument('--fontsize', type=int,
                        help='fontsize for annotations')
    parent_parser.add_argument('-xkcd', '+xkcd', action=Toggle,
                        nargs=0, help='to use the xkcd style')
    parent_parser.add_argument('--shrinkcb', type=float,
                        help='color bar shrink')
    parent_parser.add_argument('-pdf', '+pdf', action=Toggle, nargs=0,
                        help='produces non-rasterized, high quality \
                        pdf (slow!)')
    parent_parser.set_defaults(**constants.DEFAULT_CONFIG)

    # parser for the "field" command
    parser_fd = subparsers.add_parser('field', parents=[parent_parser],
                        prefix_chars='-+', help='plot scalar fields')
    parser_fd.set_defaults(func=commands.field_cmd)

    # parser for the "rprof" command
    parser_rp = subparsers.add_parser('rprof', parents=[parent_parser],
                        prefix_chars='-+', help='plot radial profiles')
    parser_rp.set_defaults(func=commands.rprof_cmd)

    # parser for the "time" command
    parser_tm = subparsers.add_parser('time', parents=[parent_parser],
                        prefix_chars='-+', help='plot temporal series')
    parser_tm.set_defaults(func=commands.time_cmd)

    # parser for the "var" command
    parser_var = subparsers.add_parser('var',
                        help='print the list of variables')
    parser_var.set_defaults(func=commands.var_cmd)

    args = main_parser.parse_args()
    if not args.func is commands.var_cmd:
        args = misc.parse_timesteps(args)
        args = misc.plot_backend(args)
    args.func(args)

if __name__ == '__main__':
    main_func()
