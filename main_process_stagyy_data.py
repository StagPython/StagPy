#!/usr/bin/env python2
"""
  Read and plot stagyy binary data
  Author: Martina Ulvrova
  Date: 2014/12/02
"""

from __future__ import print_function
import argparse
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import sys

from stag import ReadStagyyData

parser = argparse.ArgumentParser(
    description='read and process StagYY binary data')
parser.add_argument('-g', '--geometry', default='annulus',
                    choices=['annulus'],
                    help='geometry of the domain')
parser.add_argument('-p', '--path', default='./',
                    help='StagYY output directory')
parser.add_argument('-n', '--name', default='test',
                    help='StagYY generic output file name')
parser.add_argument('-s', '--timestep', default=100, type=int,
                    help='timestep')
parser.add_argument('-o', '--plot', nargs='?', const='', default='tps',
                    help='specify which variables to plot, use --var \
                    option for a list of available variables')
parser.add_argument('--var', action='store_true',
                    help='display a list of available variables')

args = parser.parse_args()

if args.var:
    print('Not implemented yet.')
    sys.exit()

dsa = 0.1  # thickness of the sticky air
verbose_figures = True  # Not clear to me what this is for
shrinkcb = 0.5

plot_temperature = 't' in args.plot
plot_pressure = 'p' in args.plot
plot_streamfunction = 's' in args.plot

#==========================================================================
# read temperature field
#==========================================================================
if plot_temperature:
    par_type = 't'
    temp = ReadStagyyData(args.path, args.name, par_type, args.timestep)
    temp_field = temp.fields[0]

    # adding a row at the end to have continuous field
    if args.geometry == 'annulus':
        newline = temp_field[:, 0, 0]
        temp_field = np.vstack([temp_field[:, :, 0].T, newline]).T
        temp.ph_coord = np.append(
            temp.ph_coord, temp.ph_coord[1] - temp.ph_coord[0])

# read concentration field
# par_type='c'
# conc=ReadStagyyData(args.path,args.name,par_type,args.timestep)

    XX, YY = np.meshgrid(
        np.array(temp.ph_coord), np.array(temp.r_coord) + temp.rcmb)

    if verbose_figures:
        fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar'))
        if args.geometry == 'annulus':
            surf = ax.pcolormesh(XX, YY, temp_field)
            cbar = plt.colorbar(
                surf, orientation='horizontal', shrink=shrinkcb, label='Temperature')
            plt.axis([temp.rcmb, np.amax(XX), 0, np.amax(YY)])

        plt.savefig(args.name + "_T.pdf", format='PDF')

        plt.show(block=False)

#==========================================================================
# read velocity-pressure field
#==========================================================================
if plot_pressure or plot_streamfunction:
    par_type = 'vp'
    vp = ReadStagyyData(args.path, args.name, par_type, args.timestep)
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
                    surf, orientation='horizontal', shrink=shrinkcb, label='Pressure')
                plt.axis([vp.rcmb, np.amax(XX), 0, np.amax(YY)])

            plt.savefig(args.name + "_p.pdf", format='PDF')

if plot_streamfunction:
    vphi = vy_field[:, :, 0]
    vph2 = -0.5 * (vphi + np.roll(vphi, 1, 1))  # interpolate to the same phi
    vr = vz_field[:, :, 0]
    nr, nph = np.shape(vr)
    stream = np.zeros(np.shape(vphi))
    # integrate first on phi
    stream[0, 1:nph - 1] = vp.rcmb * \
        integrate.cumtrapz(vr[0, 0:nph - 1], vp.ph_coord)
    stream[0, 0] = 0
    # use r coordinates where vphi is defined
    rcoord = vp.rcmb + np.array(vp.rg[0:np.shape(vp.rg)[0] - 1:2])
    for iph in range(0, np.shape(vph2)[1] - 1):
        stream[1:nr, iph] = stream[0, iph] + \
            integrate.cumtrapz(vph2[:, iph], rcoord)  # integrate on r
    stream = stream - np.mean(stream[nr / 2, :])
    # remove some typical value. Would be better to compute the golbal average
    # taking into account variable grid spacing

    vp.ph_coord = np.append(vp.ph_coord, vp.ph_coord[1] - vp.ph_coord[0])
    XX, YY = np.meshgrid(np.array(vp.ph_coord), np.array(vp.r_coord) + vp.rcmb)

    if verbose_figures:
        fig, ax = plt.subplots(ncols=1, subplot_kw=dict(projection='polar'))
        if args.geometry == 'annulus':
            surf = ax.pcolormesh(XX, YY, stream)
            cbar = plt.colorbar(
                surf, orientation='horizontal', shrink=shrinkcb, label='Stream function')
            plt.axis([vp.rcmb, np.amax(XX), 0, np.amax(YY)])

        plt.savefig(args.name + "_SF.pdf", format='PDF')

        plt.show(block=False)
