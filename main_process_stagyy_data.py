#!/bin/python
"""
  Read and plot stagyy binary data
  Author: Martina Ulvrova
  Date: 2014/12/02
"""

from numpy import *
from scipy import *
from pylab import *
from matplotlib.colors import LogNorm
from stag import ReadStagyyData

close('all')

#==========================================================================
# GENERAL SWITCHES AND DEFINITIONS
#==========================================================================
dsa=0.1 # thickness of the sticky air
verbose_figures=True # Not clear to me what this is for
shrinkcb=0.5
geometry='Annulus'

# section w/input
# These should be input parameters to this program.
ipath='/Users/stephane/Documents/Python/StagPy/'
iname='test'
iti_fn=100 # timestep 

plot_temperature=True
plot_pressure=True


#==========================================================================
# read temperature field
#==========================================================================
if plot_temperature:
    par_type='t'
    temp=ReadStagyyData(ipath,iname,par_type,iti_fn)
    temp.catch_header()
    temp.read_scalar_file()

    if geometry=='Annulus': # adding a row at the end to have continuous field
        newline=temp.field[:,0,0]
        temp.field=vstack([temp.field[:,:,0].T,newline]).T
        temp.ph_coord=append(temp.ph_coord,temp.ph_coord[1]-temp.ph_coord[0])

# read concentration field
#par_type='c'
#conc=ReadStagyyData(ipath,iname,par_type,iti_fn)
#conc.catch_header()
#conc.read_scalar_file()

    XX,YY=meshgrid(array(temp.ph_coord),array(temp.r_coord)+temp.rcmb)

    if verbose_figures:
        fig, ax=subplots(ncols=1,subplot_kw=dict(projection='polar'))
        if geometry=='Annulus':
            surf=ax.pcolormesh(XX,YY,temp.field)
            cbar=colorbar(surf,orientation='horizontal',shrink=shrinkcb,label='Temperature')
            axis([temp.rcmb,amax(XX),0,amax(YY)])

        savefig(iname + "_T.pdf",format='PDF')
   
        show(block=False)

#==========================================================================
# read velocity-pressure field
#==========================================================================
if plot_pressure:
    par_type='vp'
    vp=ReadStagyyData(ipath,iname,par_type,iti_fn)
    vp.catch_header()
    vp.read_vector_file()

    if geometry=='Annulus': # adding a row at the end to have continuous field
        newline=vp.p[:,0,0]
        vp.ph_coord=append(vp.ph_coord,vp.ph_coord[1]-vp.ph_coord[0])

        XX,YY=meshgrid(array(vp.ph_coord),array(vp.r_coord)+vp.rcmb)

    
    if verbose_figures:
        fig, ax=subplots(ncols=1,subplot_kw=dict(projection='polar'))
        if geometry=='Annulus':
            surf=ax.pcolormesh(XX,YY,vp.p[:,:,0])
            cbar=colorbar(surf,orientation='horizontal',shrink=shrinkcb,label='Pressure')
            axis([vp.rcmb,amax(XX),0,amax(YY)])

        savefig(iname + "_p.pdf",format='PDF')
