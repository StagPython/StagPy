#!/usr/local/bin/python

import numpy as np
from pylab import *
from math import pi
import matplotlib.pyplot as plt

#import sys

#print sys.path

import f90nml
import os

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

# Simple mouse click function to store coordinates
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    button = event.button

    # print 'x = %d, y = %d'%(
    #     ix, iy)

    # assign global variable to access outside of function
    global coords
    if button==3:
        coords.append((ix, iy))

    # Disconnect after 1 clicks
    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)
        plt.close(1)
    return

test = raw_input('Compute statistics? [Y/n] ')
compstat = test=='y' or not test

plt.style.use('ggplot')

"""Read par file in the parent or present directory. Should be optional or tested"""
read_par_file=True
if os.path.exists('../par'):
    par_file = '../par'
elif os.path.exists('par'):
    par_file = 'par'
else:
    print 'No par file found. Input pars by hand'
    read_par_file=False
    nz = int(raw_input('nz = '))
    rcmb=1
    geom=str(raw_input('spherical (s) or cartesian (c)? '))
    Spherical=geom=='s'

print 'par file = ',par_file
    
if read_par_file:
    nml = f90nml.read(par_file)
    Spherical=nml['geometry']['shape']=='Spherical' or nml['geometry']['shape']=='spherical'
    if Spherical:
        rcmb=nml['geometry']['r_cmb']
    else:
        rcmb=1
    nz=nml['geometry']['nztot']
    stem = nml['ioin']['output_file_stem']
    rep, sep, s =  stem.partition('/')
    timefile = s+'_time.dat'
    Ra=nml['refstate']['Ra0']
    Rh=nml['refstate']['Rh']
    BotPphase=nml['boundaries']['BotPphase']
    
print 'Spherical =',Spherical
if Spherical:
    print 'r_cmb =',rcmb
print 'nz =',nz

with open(timefile, 'r') as infile:
    first = infile.readline()

colnames = first.split()
# suppress two columns from the header. Only temporary since this has been corrected in stag
if len(colnames)==33:
    colnames = colnames[:28]+colnames[30:]

data = np.loadtxt(timefile,skiprows=1)

nt = len(data)
#print 'nt = ',nt

if Spherical:
    rb=rcmb
    rs=rb+1
    coefb = 1 #rb**2*4*pi
    coefs = (rs/rb)**2 #*4*pi
    volume = rb*(1-(rs/rb)**3)/3 #*4*pi/3
else:
    coefb=1
    coefs=1
    volume=1

t=data[:,1]
    
dTdt = (data[2:nt-1,5]-data[0:nt-3,5])/(data[2:nt-1,1]-data[0:nt-3,1])
Ebalance = data[1:nt-2,2]*coefs - data[1:nt-2,3]*coefb - volume*dTdt

fig=plt.figure(figsize=(30,10))

plt.subplot(2, 1, 1)
plt.plot(t, data[:,2]*coefs,'b', label='Surface')
plt.plot(t, data[:,3]*coefb,'r', label='Bottom')
plt.plot(t[1:nt-2:], Ebalance,'g', label='Energy balance')
plt.ylabel('Heat flow')
plt.legend = plt.legend(loc='upper right', shadow=False, fontsize='x-large')
plt.legend.get_frame().set_facecolor('white')
    
plt.subplot(2, 1, 2)
plt.plot(t, data[:,5],'k')
plt.xlabel('Time')
plt.ylabel('Mean temperature')

plt.savefig("flux_time.pdf",format='PDF')

if compstat:
    coords = []
    print 'right click to select starting time of statistics computations'
# Call click func
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()


moy = []
rms = []
Ebal = []
rmsEbal =[]
if compstat:
    ch1 = np.where(t == (find_nearest(t, coords[0][0])))

    print 'Statistics computed from t ='+str(t[ch1[0][0]])
    for num in range(2,len(colnames)):
        moy.append(trapz(data[ch1[0][0]:nt-1,num], x = t[ch1[0][0]:nt-1])/(t[nt-1]-t[ch1[0][0]]))
        rms.append(sqrt(trapz((data[ch1[0][0]:nt-1,num]-moy[num-2])**2, x = t[ch1[0][0]:nt-1])/(t[nt-1]-t[ch1[0][0]])))
        print colnames[num]+' = '+str(moy[num-2])+' \pm '+str(rms[num-2])
    Ebal.append(trapz(Ebalance[ch1[0][0]-1:nt-3], x = t[ch1[0][0]:nt-2])/(t[nt-2]-t[ch1[0][0]]))
    rmsEbal.append(sqrt(trapz((Ebalance[ch1[0][0]-1:nt-3]-Ebal)**2, x = t[ch1[0][0]:nt-2])/(t[nt-2]-t[ch1[0][0]])))
    print 'Energy balance '+str(Ebal)+' \pm '+str(rmsEbal) 
    results=moy+Ebal+rms+rmsEbal
    fich=open('Stats.dat','w')
    fich.write("%10.5e %10.5e %10.5e " % (Ra, Rh, BotPphase))
    for item in results:
        fich.write("%10.5e " % item)
    fich.write("\n")
    fich.close()


      #      np.savetxt('Stats.dat',np.array(results),delimiter=' ',newline=' ')

