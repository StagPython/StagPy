#!/usr/bin/env python2
"""
  Read and plot stagyy binary data
  Author: Martina Ulvrova
  Date: 2014/12/02
"""

from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys

from stag import StagyyData
import constants
import misc

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

verbose_figures = True  # Not clear to me what this is for

plot_temperature = 't' in args.plot
plot_pressure = 'p' in args.plot
plot_streamfunction = 's' in args.plot

#==========================================================================
# read temperature field
#==========================================================================
if plot_temperature:
    par_type = 't'
    temp = StagyyData(args, par_type)
    temp_field = temp.fields[0]

    # adding a row at the end to have continuous field
    if args.geometry == 'annulus':
        newline = temp_field[:, 0, 0]
        temp_field = np.vstack([temp_field[:, :, 0].T, newline]).T
        temp.ph_coord = np.append(
            temp.ph_coord, temp.ph_coord[1] - temp.ph_coord[0])

# read concentration field
# par_type='c'
# conc=StagyyData(args, par_type)

    XX, YY = np.meshgrid(
        np.array(temp.ph_coord), np.array(temp.r_coord) + temp.rcmb)

    if verbose_figures:
        fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar'))
        if args.geometry == 'annulus':
            surf = ax.pcolormesh(XX, YY, temp_field)
            cbar = plt.colorbar(
                surf, orientation='horizontal',
                shrink=args.shrinkcb, label='Temperature')
            plt.axis([temp.rcmb, np.amax(XX), 0, np.amax(YY)])

        plt.savefig(args.name + "_T.pdf", format='PDF')

        plt.show(block=False)

#==========================================================================
# read velocity-pressure field
#==========================================================================
if plot_pressure or plot_streamfunction:
    par_type = 'vp'
    vp = StagyyData(args, par_type)
    vx_field = vp.fields[0]
    vy_field = vp.fields[1]
    vz_field = vp.fields[2]
    p_field = vp.fields[3]

    if plot_pressure:
        # adding a row at the end to have continuous field
        if args.geometry == 'annulus':
            newline = p_field[:, 0, 0]
            vp.ph_coord_new = np.append(
                vp.ph_coord, vp.ph_coord[1] - vp.ph_coord[0])

        XX, YY = np.meshgrid(
            np.array(vp.ph_coord_new), np.array(vp.r_coord) + vp.rcmb)

        if verbose_figures:
            fig, ax = plt.subplots(
                ncols=1, subplot_kw=dict(projection='polar'))
            if args.geometry == 'annulus':
                surf = ax.pcolormesh(XX, YY, p_field[:, :, 0])
                cbar = plt.colorbar(
                    surf, orientation='horizontal',
                    shrink=args.shrinkcb, label='Pressure')
                plt.axis([vp.rcmb, np.amax(XX), 0, np.amax(YY)])

            plt.savefig(args.name + "_p.pdf", format='PDF')

if plot_streamfunction:
    stream = misc.calc_stream(vp)
    vp.ph_coord = np.append(vp.ph_coord, vp.ph_coord[1] - vp.ph_coord[0])
    XX, YY = np.meshgrid(np.array(vp.ph_coord), np.array(vp.r_coord) + vp.rcmb)

    if verbose_figures:
        fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar'))
        if args.geometry == 'annulus':
            surf = ax.pcolormesh(XX, YY, stream)
            cbar = plt.colorbar(
                surf, orientation='horizontal', shrink=args.shrinkcb,
                label='Stream function')
            plt.axis([vp.rcmb, np.amax(XX), 0, np.amax(YY)])

        plt.savefig(args.name + "_SF.pdf", format='PDF')

        plt.show(block=False)
