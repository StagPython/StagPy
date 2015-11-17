"""
  plots radial profiles coming out of stagyy
  Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
  Date: 2015/09/11
"""

#from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import f90nml
import os
import sys
import misc
import seaborn as sns

def rprof_cmd(args):
    '''
    Function to plot radial profiles
    '''
    istart, ilast, istep = args.timestep

    ############### INPUT PARAMETERS ################
    # should rather be default parameters to be replaced by command line options

    cmap = plt.cm.gist_ncar
    #cmap = plt.cm.Accent
    #cmap = plt.cm.gist_earth
    #cmap = plt.colors.Colormap
    #plt.style.use('ggplot')
    #cmap=green_purple = brewer2mpl.get_map('PRGn', 'diverging', 11).mpl_colormap

    plot_grid = True
    plot_temperature = True
    plot_minmaxtemp = True
    plot_velocity = True
    plot_minmaxvelo = False
    plot_viscosity = False
    plot_minmaxvisco = False
    plot_advection = True
    plot_energy = True
    plot_concentration = True
    plot_minmaxcon = True
    plot_conctheo = False

    lwdth = 2
    ftsz = 16
    # parameters for the theoretical composition profiles
    Rmin = 1.19
    Rmax = Rmin + 1.
    xi0l = 0.1
    DFe = 0.85
    xi0s = DFe*xi0l
    xieut = 0.8

    #"""Read par file in the parent or present directory. Should be optional or tested"""
    # should be a separated func in misc module
    read_par_file = True
    if os.path.exists('../par'):
        par_file = '../par'
    elif os.path.exists('par'):
        par_file = 'par'
    else:
        print 'No par file found. Input pars by hand'
        read_par_file = False
        nz = int(raw_input('nz = '))
        rcmb = 1
        geom = str(raw_input('spherical (s) or cartesian (c)? '))
        Spherical = geom == 's'

    if read_par_file:
        nml = f90nml.read(par_file)
        Spherical = nml['geometry']['shape'] == 'Spherical'
        if Spherical:
            rcmb = nml['geometry']['r_cmb']
        nz = nml['geometry']['nztot']
        stem = nml['ioin']['output_file_stem']
        ste = stem.split('/')[-1]
        if os.path.exists('../'+ste+'_rprof.dat'):
            proffile = '../'+ste+'_rprof.dat'
        elif os.path.exists(ste+'_rprof.dat'):
            proffile = ste+'_rprof.dat'
        else:
            print 'No profile file found.'
        Rmin = rcmb
        Rmax = rcmb+1
        if 'fe_eut' in nml['tracersin']:
            xieut = nml['tracersin']['fe_eut']
        if 'k_fe' in nml['tracersin']:
            DFe = nml['tracersin']['k_fe']
        if 'fe_cont' in nml['tracersin']:
            xi0l = nml['tracersin']['fe_cont']

    xi0s = DFe*xi0l
    xired = xi0l/xieut
    Rsup = (Rmax**3-xired**(1/(1-DFe))*(Rmax**3-Rmin**3))**(1./3.)

    def initprof(rpos):
        '''Theoretical profile at the end of magma ocean crystallization'''
        if rpos < Rsup:
            return xi0s*((Rmax**3-Rmin**3)/(Rmax**3-rpos**3))**(1-DFe)
        else:
            return xieut

    pi = np.pi
    if Spherical:
        rb = rcmb
        rs = rb+1
        coefb = rb**2*4*pi
        coefs = rs**2*4*pi
    else:
        coefb = 1
        coefs = 1

    timesteps = []
    data0 = []
    lnum = -1
    fich = open(proffile)
    for line in fich:
        if line != '\n':
            lnum = lnum+1
            lll = ' '.join(line.split())
            if line[0] == '*':
                timesteps.append([lnum, int(lll.split(' ')[1]), float(lll.split(' ')[5])])
            else:
                llf = np.array(lll.split(' '))
                data0.append(llf)

    tsteps = np.array(timesteps)
    nsteps = tsteps.shape[0]
    data = np.array(data0)
    if ilast == -1:
        ilast = nsteps-1
    if istart == -1:
        istart = nsteps-1

    nzp = []
    for it in range(0, nsteps-1):
        nzp = np.append(nzp, tsteps[it+1, 0]-tsteps[it, 0]-1)

    nzp = np.append(nzp, lnum-tsteps[nsteps-1, 0])

    nzs = [[0, 0, 0]]
    nzc = 0
    for it in range(1, nsteps):
        if nzp[it] != nzp[it-1]:
            nzs.append([it, it-nzc, int(nzp[it-1])])
            nzc = it
    if nzp[nsteps-1] != nzs[-1][1]:
        nzs.append([nsteps, nsteps-nzc, int(nzp[nsteps-1])])

    nzi = np.array(nzs)

    num_plots = np.floor((nsteps-istart)/istep)+1
    icol = np.linspace(0, 1, num_plots)

    def plotprofiles(quant, *vartuple):
        """Plot the chosen profiles for the chosen timesteps

        vartuple contains the numbers of the column to be plotted

        A kwarg should be used for different options, e.g. whether
        densities of total values are plotted
        """
        if quant == 'Grid':
            fig, ax = plt.subplots(2, sharex=True)
        else:
            fig = plt.figure()
        for step in range(istart, ilast, istep):
            step = step +1# starts at 0=> 15 is the 16th
    #       print step
            an = sorted(np.append(nzi[:, 0], step))
            inn = an.index(step)

            nnz = np.multiply(nzi[:, 1], nzi[:, 2])

            i0 = np.sum([nnz[0:inn]])+(step-nzi[inn-1, 0]-1)*nzi[inn, 2]
            i1 = i0+nzi[inn, 2]-1
            '''Plot the chosen profiles'''
            jj = 0
            if quant == 'Grid':
                ax[0].plot(data[i0:i1, 0], '-ko', label='z')
                ax[0].set_ylabel('z', fontsize=ftsz)

                dz = np.array(data[i0+1:i1, 0], np.float) - np.array(data[i0:i1-1, 0], np.float)
                ax[1].plot(dz, '-ko', label='dz')
                ax[1].set_xlabel('cell number', fontsize=ftsz)
                ax[1].set_ylabel('dz', fontsize=ftsz)
            else:

                for j in vartuple:
                    if jj == 0:
                        snsplot = sns.plt(data[i0:i1, j], data[i0:i1, 0], linewidth=lwdth, label=r'$t=%.2e$' % (tsteps[step-1, 2]))
                        col = snsplot[0].get_color()
                        if (quant == 'Concentration' or quant == 'Temperature') and plot_conctheo and step == istart+1:
                            ri = np.array(data[i0:i1, 0], np.float)+Rmin
                            rf = (Rmax**3.+Rmin**3.-ri**3.)**(1./3.)-Rmin
                            sns.plt(data[i0:i1, j], rf, 'b--', linewidth=lwdth, label='Overturned')
                        jj = 1
                    else:
                        if jj == 1:
                            lstyle = '--'
                        elif jj == 2:
                            lstyle = ':'
                        else:
                            lstyle = '-.'
                        sns.plt(data[i0:i1, j], data[i0:i1, 0], c=col, linestyle=lstyle, linewidth=lwdth)
                        jj = jj+1
                plt.ylim([-0.1, 1.1])

                plt.xlabel(quant, fontsize=ftsz)
                plt.ylabel('z', fontsize=ftsz)
                plt.xticks(fontsize=ftsz)
                plt.yticks(fontsize=ftsz)
        if quant == 'Grid':
            plt.savefig("Grid.pdf", format='PDF')
        else:
            lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                    borderaxespad=0., mode="expand",
                    ncol=3, fontsize=ftsz,
                    columnspacing=1.0, labelspacing=0.0,
                    handletextpad=0.0, handlelength=1.5,
                    fancybox=True, shadow=False)

            plt.savefig(quant.replace(' ', '_')+".pdf", format='PDF', bbox_extra_artists=(lgd, ), bbox_inches='tight')

        return

    '''Now use it for the different types of profiles'''

    if plot_temperature:
        if plot_minmaxtemp:
            plotprofiles('Temperature', 1, 2, 3)
        else:
            plotprofiles('Temperature', 1)

    if plot_velocity:
        if plot_minmaxvelo:
            plotprofiles('Vertical Velocity', 7, 8, 9)
        else:
            plotprofiles('Vertical Velocity', 7)

    if plot_viscosity:
        if plot_minmaxvisco:
            plotprofiles('Viscosity', 13, 14, 15)
        else:
            plotprofiles('Viscosity', 13)

    if plot_concentration:
        if plot_minmaxcon:
            plotprofiles('Concentration', 36, 37, 38)
        else:
            plotprofiles('Concentration', 36)

    '''Plot grid spacing'''
    if plot_grid:
        plotprofiles('Grid')

    '''Plot advection profiles'''
    if plot_advection:
        plotprofiles('Advection', 57, 58, 59)

    sys.exit()


    # Plot the profiles of vertical advection: total and contributions from up-
    # and down-welling currents

    # if plot_advection:
    #     figure()
    #     if Spherical:
    #         f, (ax1, ax2) = subplots(1, 2, sharey=True)
    #         ax1.plot(data[:,57],data[:,0], '-ko', label='Total')
    #         ax1.plot(data[:,58],data[:,0], '-bo', label='Down')
    #         ax1.plot(data[:,59],data[:,0], '-ro', label='Up')
    #         if data.shape[1]>63:
    #             ax1.plot(data[:,60],data[:,63], 'kx')
    #             ax1.plot(data[:,61],data[:,63], 'bx')
    #             ax1.plot(data[:,62],data[:,63], 'rx')
    #         ylim([-0.1,1.1])
    #         ax1.set_xlabel("Advection per unit surface",fontsize=12)
    #         ax1.set_ylabel("z",fontsize=12)
    #         ax1.legend(loc='upper right', shadow=False, fontsize='x-large')
    # #    ax1.legend.set_facecolor('white')

    #         ax2.plot(data[:,57]*(data[:,0]+rcmb)**2,data[:,0], '-ko', label='Total')
    #         ax2.plot(data[:,58]*(data[:,0]+rcmb)**2,data[:,0], '-bo', label='Down')
    #         ax2.plot(data[:,59]*(data[:,0]+rcmb)**2,data[:,0], '-ro', label='Up')
    #         if data.shape[1]>63:
    #             ax2.plot(data[:,60]*(data[:,63]+rcmb)**2,data[:,63], 'kx',label='Total, vz points')
    #             ax2.plot(data[:,61]*(data[:,63]+rcmb)**2,data[:,63], 'bx')
    #             ax2.plot(data[:,62]*(data[:,63]+rcmb)**2,data[:,63], 'rx')
    #         ax2.set_xlabel("Total advection",fontsize=12)
    #     else:
    #         plot(data[:,57],data[:,0], 'ko', label='Total')
    #         plot(data[:,58],data[:,0], '-bo', label='Down')
    #         plot(data[:,59],data[:,0], '-ro', label='Up')
    #         if data.shape[1]>63:
    #             plot(data[:,60],data[:,63], '-kx',label='Total, vz points')
    #             plot(data[:,61],data[:,63], 'bx')
    #             plot(data[:,62],data[:,63], 'rx')
    #         ylim([-0.1,1.1])
    #         xlabel("Advection per unit surface",fontsize=12)
    #         ylabel("z",fontsize=12)
    #         legend = plt.legend(loc='upper right', shadow=False, fontsize='x-large')
    #         legend.get_frame().set_facecolor('white')

    #     savefig("Adv_prof.pdf",format='PDF')

    # # Plot the energy balance as a function of depth
    # if plot_energy:
    #     z=data[:,63]
    #     dz = data[1:nz,0]-data[0:nz-1,0]
    #     qcond = (data[0:nz-1,1]-data[1:nz,1])/dz
    #     qadv=data[:,60]
    #     qcond0=(1-data[0,1])/data[0,0]
    #     qtop=data[nz-1,1]/(1-data[nz-1,0])
    #     qcond=insert(qcond,0,qcond0)
    #     qcond=append(qcond,qtop)
    #     z=append(z,1)
    #     qadv=append(qadv,0)
    #     qtot=qadv+qcond
    #     dz=z[1:nz+1]-z[0:nz]
    #     dqadv=(qadv[1:nz+1]-qadv[0:nz])/dz
    #     dqcond=(qcond[1:nz+1]-qcond[0:nz])/dz

    #     figure()
    #     if Spherical:
    #         plot(qadv*(z+rcmb)**2,z,'-ko',label='Advection')
    #         plot(qcond*(z+rcmb)**2,z,'-bo',label='Conduction')
    #         plot(qtot*(z+rcmb)**2,z,'-ro',label='Total')       
    #         xlabel("Integrated heat flow",fontsize=12)
    #     else:
    #         plot(qadv,z,'-ko',label='Advection')
    #         plot(qcond,z,'-bo',label='Conduction')
    #         plot(qtot,z,'-ro',label='Total')
    #         xlabel("Heat flux",fontsize=12)
    #     ylim([-0.1,1.1])
    #     ylabel("z",fontsize=12)
    #     legend = plt.legend(loc='best', shadow=False, fontsize='x-large')
    #     legend.get_frame().set_facecolor('white')
    #     savefig("Energy_prof.pdf",format='PDF')
    #     dzopt=(1-data[0,1])/data[0,57]
    #     print 'dz for energy balance in steady state : ',dzopt
    #     print 'actual dz = ',data[0,0]
    #  #   print qcond.shape,qadv.shape,z.shape
    # print dqcond.shape, data[:,0].shape
    # figure()
    # plot(data[:,31],data[:,0],'-ko',label='advection')
    # plot(-dqadv,data[:,0],'kx')
    # plot(data[:,32],data[:,0],'-bo',label='conduction')
    # plot(-dqcond,data[:,0],'bx')
    # plot(data[:,31]+data[:,32]+data[:,33],data[:,0],'-rx',label='total')

    # legend = plt.legend(loc='best', shadow=False, fontsize='x-large')
    # savefig("Adv_prof2.pdf",format='PDF')
