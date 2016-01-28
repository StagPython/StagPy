import numpy as np
import sys
from .stagdata import BinData, RprofData

def detectPlates(stagdat_t,stagdat_vp,rprof_data,args):
    Vz=stagdat_vp.fields['w']
    Vphi=stagdat_vp.fields['v']
    Tcell=stagdat_t.fields['t']
    data, tsteps, nzi = rprof_data.data, rprof_data.tsteps, rprof_data.nzi
    nz=len(Vz)
    nphi=len(Vz[0])-1
    radius=list(map(float,data[0:nz,0]))
    spherical = args.par_nml['geometry']['shape'].lower() == 'spherical'
    if spherical:
        rcmb = args.par_nml['geometry']['r_cmb']
    else:
        rcmb = 0.
    rmin = rcmb
    rmax = rcmb + 1.
    dphi=1/nphi
    
    #calculing radius on the grid
    radiusgrid = len(radius)*[0]
    radiusgrid+=[1]
    for i in range(1,len(radius)):
        radiusgrid[i]=2*radius[i-1]-radiusgrid[i-1]
    for i in range(len(radiusgrid)):
        radiusgrid[i]+=rcmb
    for i in range(len(radius)):
        radius[i]+=rcmb
        
    #calculing Tmean
    Tmean=0
    for r in range(len(radius)):
        for phi in range(nphi):
            Tmean+=(radiusgrid[r+1]**2-radiusgrid[r]**2)*dphi*Tcell[r,phi]
    Tmean = Tmean/(radiusgrid[-1]**2-rcmb**2)
    
    #calculing temperature on the grid
    Tgrid=np.zeros((nz+1,nphi))
    for phi in range(nphi):
        Tgrid[0,phi]=1
    for z in range(1,nz):
        for phi in range(nphi):
            Tgrid[z,phi]=(Tcell[z-1,phi]*(radiusgrid[z]-radius[z-1])+Tcell[z,phi]*(-radiusgrid[z]+radius[z]))/(radius[z]-radius[z-1])
    
    flux_c=nz*[0]
    for z in range(1,nz-1):
        for phi in range(nphi):
            flux_c[z]+=(Tgrid[z,phi]-Tmean)*Vz[z,phi,0]*radiusgrid[z]*dphi
    
    #checking stagnant lid
    bool=True
    max=np.max(flux_c)
    for z in range(nz-int(nz/20),nz):
        if abs(flux_c[z]) > max/50:
            bool = False
    if bool:
        print('stagnant lid')
        sys.exit()
    else:
        #verifying horizontal plate speed
        dVphi=nphi*[0]
        for phi in range(0,nphi):
            dVphi[phi]=(Vphi[nz-1,phi,0]-Vphi[nz-1,phi-1,0])/((1+rcmb)*dphi)
        limits=[]
        max_dVphi=0
        for i in dVphi:
            if abs(i)>max_dVphi:
                max_dVphi=abs(i)
        seuilVphi=max_dVphi/30
        for phi in range(0,nphi-1):
            if abs(dVphi[phi])>=abs(dVphi[phi-1]) and abs(dVphi[phi])>=abs(dVphi[phi+1]) and abs(dVphi[phi])>=seuilVphi:
                limits+=[phi]
        if abs(dVphi[nphi-1])>=abs(dVphi[nphi-2]) and abs(dVphi[nphi-1])>=abs(dVphi[0]) and abs(dVphi[nphi-1])>=seuilVphi:
            limits+=[nphi-1]
        print(limits)
        
        #verifying closeness of limits
        while abs(nphi-limits[-1]+limits[0])<=(nphi/80):
            newlimit=(-nphi+limits[-1]+limits[0])/2
            if newlimit<0:
                newlimit=nphi+newlimit
            limits=[newlimit]+limits[1:-1]
            limits.sort()
        i=1
        while i < len(limits):
            if abs(limits[i-1]-limits[i])<=(nphi/80):
                limits=limits[:i-1]+[(limits[i-1]+limits[i])/2]+limits[i+1:]
            else:
                i+=1
        
        print(limits)
        #verifying vertical speed
        for phi in limits:
            Vzm=0
            if phi == nphi-1:
                for z in range(int(4*nz/5),nz):
                    Vzm+=(abs(Vz[z,phi,0])+abs(Vz[z,phi-1,0])+abs(Vz[z,0,0]))*5/(nz*3)
            else:
                for z in range(int(4*nz/5),nz):
                    Vzm+=(abs(Vz[z,phi,0])+abs(Vz[z,phi-1,0])+abs(Vz[z,phi+1,0]))*5/(nz*3)
            seuilVz=np.max(Vz[int(nz/3),:,0])/10
            if Vzm<seuilVz:
                limits.remove(phi)
                
        print(limits)
        for i in range(len(limits)):
            limits[i]=int(limits[i])
    return(limits,nphi,dVphi)

def plates_cmd(args):
    """plates analysis"""
    stgdat_vp= BinData(args,'p',args.timestep[0])
    stgdat_t= BinData(args,'t',args.timestep[0])
    rprof_data=RprofData(args)
    plt=args.plt
    limits, nphi,dVphi = detectPlates(stgdat_t,stgdat_vp,rprof_data,args)
    limits.sort()
    sizePlates=[limits[0]+nphi-limits[-1]]
    for l in range(1,len(limits)):
        sizePlates+=[limits[l]-limits[l-1]]
    plt.hist(sizePlates,10,(0,nphi/2))
    
        
        
