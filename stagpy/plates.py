import numpy as np
import sys
from .stagdata import BinData, RprofData

def detectPlates(stagdat,rprof_data,args):
    Vz=stagdat.fields['w']
    Vphi=stagdat.fields['v']
    data, tsteps, nzi = rprof_data.data, rprof_data.tsteps, rprof_data.nzi
    fluxC=data[:,57]
    radius=data[:,0]
    
    spherical = args.par_nml['geometry']['shape'].lower() == 'spherical'
    if spherical:
        rcmb = args.par_nml['geometry']['r_cmb']
    else:
        rcmb = 0.
    rmin = rcmb
    rmax = rcmb + 1.
    integ = lambda f, r: f*(r/rmax)**2
    fluxC = map(integ,fluxC,radius)
    nz=len(Vz)
    nphi=len(Vz[0])
        
    #checking stagnant lid
    bool=True
    max=np.max(l)
    for z in range(nz-int(nz/20),nz):
        if abs(fluxC) > max/50:
            bool = False
    if bool:
        print('stagnant lid')
        sys.exit()
    else:
        #verifying horizontal plate speed
        dVphi=nphi*[0]
        for phi in range(0,nphi):
            dVphi[phi]=(Vphi[phi,nz]-Vphi[phi-1,nz])/dphi
        limits=[]
        for i in range(0,nphi):
            if abs(dVphi[phi])>=abs(dVphi[phi-1]) and abs(dVphi[phi])>=abs(dVphi[phi+1]) and abs(dVphi[phi])>=seuilVphi:
                limits+=[phi]
        #verifying vertical speed
        for i in range(len(limits)):
            if abs(limits[i-1]-limit[i])<(nphi/80):
                limits=limits[0:i-2]+(limits[i-1]+limits[i+1])/2+limits[i+1,-1]
        for phi in limits:
            Vzm=0
            for z in range(int(nz/3),nz):
                Vzm+=abs(Vz(phi,z))
            if Vzm<seuilVz:
                limits.remove(phi)
    return limits

def plates_cmd(args):
    """plates analysis"""

    stgdat= BinData(args,'p',args.timestep[0])
    rprof_data=RprofData(args)
    plt=args.plt
    limits = detectPlates(stgdat,rprof_data,args)
    limits.sort()
    sizePlates=[limits[0]+nphi-limits[-1]]
    for l in range(1,len(limits)):
        sizePlates+=[limits[l]-limits[l-1]]
    plt.hist(sizePlates,10,(0,nphi/2))
    
        
        