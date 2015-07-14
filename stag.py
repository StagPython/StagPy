#!/bin/python
import struct
import numpy as np

class ReadStagyyData:
    def __init__(self,fpath,fname,par_type,ti_fn):
        self.fpath=fpath
        self.fname=fname
        self.par_type=par_type
        self.ti_fn=ti_fn
        self.file_format='l'

        if ti_fn<10000:
            mylist=list(str(10000+ti_fn))
            mylist[0]='0'
            number_string=''.join(mylist)
        else:
            number_string=str(ti_fn)
        self.fullname=fpath+fname+'_'+par_type+number_string    #name of the file to read

        if (par_type=='t') or (par_type=='eta') or (par_type=='rho') or (par_type=='str') or (par_type=='age'):
            self.nval=1
        elif par_type=='vp':
            self.nval=4
        
    #==========================================================================
    # READ HEADER
    #==========================================================================
    def catch_header(self):
        fid=open(self.fullname,'rb') # open file
        self.byte_offset=0
        self.nmagic=struct.unpack('i',fid.read(4))[0] # Version 
        self.byte_offset=self.byte_offset+4

        if (self.nmagic<100 and self.nval>1) or (self.nmagic>300 and self.nval==1):  # check nb components
                   error('wrong number of components in field')

        nnmagic = self.nmagic % 100
        if nnmagic>=9 and self.nval==4:
             self.xyp = 1         # extra ghost point in horizontal direction
        else:
             self.xyp = 0

        self.nthtot = struct.unpack('i',fid.read(4))[0] # total number of values in the latitude direction
        self.byte_offset=self.byte_offset+4
        self.nphtot = struct.unpack('i',fid.read(4))[0] #total number of values in the longitude direction
        self.byte_offset=self.byte_offset+4
        self.nrtot       = struct.unpack('i',fid.read(4))[0] #total number of values in the radius direction
        self.byte_offset=self.byte_offset+4
        self.nblocks     = struct.unpack('i',fid.read(4))[0] # of blocks, 2 for yinyan
        self.byte_offset=self.byte_offset+4
        self.aspect=np.zeros(2)
        self.aspect[0]      = struct.unpack('f',fid.read(4))[0] #Aspect ratio
        self.aspect[1]      = struct.unpack('f',fid.read(4))[0] #Aspect ratio
        self.byte_offset=self.byte_offset+4*2;
        self.nnth         =  struct.unpack('i',fid.read(4))[0] #Number of parallel subdomains
        self.byte_offset=self.byte_offset+4
        self.nnph         =  struct.unpack('i',fid.read(4))[0] #in the th,ph,r and b directions
        self.byte_offset=self.byte_offset+4
        self.nnr         =  struct.unpack('i',fid.read(4))[0] 
        self.byte_offset=self.byte_offset+4
        self.nnb         =  struct.unpack('i',fid.read(4))[0] 
        self.byte_offset=self.byte_offset+4

        self.nr2         = self.nrtot*2+1
        self.rg          = struct.unpack('f'*self.nr2,fid.read(4*self.nr2))  # r-coordinates
        self.byte_offset=self.byte_offset+4*self.nr2

        self.rcmb        = struct.unpack('f',fid.read(4))[0] #radius of the cmb
        self.byte_offset=self.byte_offset+4
        self.ti_step     = struct.unpack('i',fid.read(4))[0]
        self.byte_offset=self.byte_offset+4
        self.ti_ad       = struct.unpack('f',fid.read(4))[0] 
        self.byte_offset=self.byte_offset+4
        self.erupta_total= struct.unpack('f',fid.read(4))[0] 
        self.byte_offset=self.byte_offset+4
        self.botT_val    = struct.unpack('f',fid.read(4))[0] 
        self.byte_offset=self.byte_offset+4


        self.th_coord           = struct.unpack('f'*self.nthtot,fid.read(4*self.nthtot)) # th-coordinates
        self.byte_offset=self.byte_offset+4*self.nthtot
        self.ph_coord           =struct.unpack('f'*self.nphtot,fid.read(4*self.nphtot)) # ph-coordinates
        self.byte_offset=self.byte_offset+4*self.nphtot
        self.r_coord           = struct.unpack('f'*self.nrtot,fid.read(4*self.nrtot)) # r-coordinates
        self.byte_offset=self.byte_offset+4*self.nrtot

        fid.close()

    #==========================================================================
    # READ SCALAR FILE
    #==========================================================================
    def read_scalar_file(self):
        fid=open(self.fullname,'rb') # open file
        fid.seek(self.byte_offset)

        # compute nth, nph, nr and nb PER CPU
        nth  =self.nthtot/self.nnth
        nph  =self.nphtot/self.nnph
        nr  =self.nrtot/self.nnr
        nb  =self.nblocks/self.nnb
        npi =(nth+self.xyp)*(nph+self.xyp)*nr*nb*self.nval #the number of values per 'read' block

        field_3D = np.zeros((self.nblocks,self.nrtot,self.nphtot,self.nthtot));

        # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        fileContent = struct.unpack('f'*npi,fid.read(4*npi))
                        data_CPU = np.array(fileContent)
                        # Create a 3D matrix from these data
                        #data_CPU_3D = data_CPU.reshape((nth,nph,nr,nb))
                        #data_CPU_3D = data_CPU.reshape((nr,nph,nth,nb))
                        data_CPU_3D = data_CPU.reshape((nb,nr,nph,nth))
                        # Add local 3D matrix to global matrix
                        sth=ithc*nth
                        eth=ithc*nth+nth
                        sph=iphc*nph
                        eph=iphc*nph+nph
                        sr=irc*nr
                        er=irc*nr+nr
                        snb=ibc*nb
                        enb=ibc*nb+nb
                        field_3D[snb:enb,sr:er,sph:eph,sth:eth] = data_CPU_3D

        self.field=field_3D[0,:,:,:]
        fid.close()
        
    #==========================================================================
    # READ VECTOR FILE
    #==========================================================================
    def read_vector_file(self):
        fid=open(self.fullname,'rb') # open file
        fid.seek(self.byte_offset)
 
        # compute nth, nph, nr and nb PER CPU
        nth  =self.nthtot/self.nnth
        nph  =self.nphtot/self.nnph
        nr  =self.nrtot/self.nnr
        nb  =self.nblocks/self.nnb
        npi =(nth+self.xyp)*(nph+self.xyp)*nr*nb*self.nval #the number of values per 'read' block

        self.scalefac= struct.unpack('f',fid.read(4))[0]

        vx_3D = np.zeros((self.nblocks,self.nrtot,self.nphtot+self.xyp,self.nthtot+self.xyp));
        vy_3D = np.zeros((self.nblocks,self.nrtot,self.nphtot+self.xyp,self.nthtot+self.xyp));
        vz_3D = np.zeros((self.nblocks,self.nrtot,self.nphtot+self.xyp,self.nthtot+self.xyp));
        p_3D = np.zeros((self.nblocks,self.nrtot,self.nphtot+self.xyp,self.nthtot+self.xyp));

                # loop over parallel subdomains
        for ibc in np.arange(self.nnb):
            for irc in np.arange(self.nnr):
                for iphc in np.arange(self.nnph):
                    for ithc in np.arange(self.nnth):
                        # read the data for this CPU
                        fileContent = struct.unpack('f'*npi,fid.read(4*npi))
                        data_CPU = np.array(fileContent)
                        data_CPU = data_CPU*self.scalefac
                        # Create a 3D matrix from these data
                        data_CPU_3D = data_CPU.reshape((nb,nr,nph+self.xyp,nth+self.xyp,self.nval))
                        # Add local 3D matrix to global matrix
                        sth=ithc*nth
                        eth=ithc*nth+nth+self.xyp
                        sph=iphc*nph
                        eph=iphc*nph+nph+self.xyp
                        sr=irc*nr
                        er=irc*nr+nr
                        snb=ibc*nb
                        enb=ibc*nb+nb
                        vx_3D[snb:enb,sr:er,sph:eph,sth:eth]=data_CPU_3D[:,:,:,:,0]
                        vy_3D[snb:enb,sr:er,sph:eph,sth:eth]=data_CPU_3D[:,:,:,:,1]
                        vz_3D[snb:enb,sr:er,sph:eph,sth:eth]=data_CPU_3D[:,:,:,:,2]
                        p_3D[snb:enb,sr:er,sph:eph,sth:eth]=data_CPU_3D[:,:,:,:,3]

        self.vx=vx_3D[0,:,:,:]
        self.vy=vy_3D[0,:,:,:]
        self.vz=vz_3D[0,:,:,:]
        self.p=p_3D[0,:,:,:]
        fid.close()

        
