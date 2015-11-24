"""
ow  plots radial profiles coming out of stagyy
  Author: Stephane Labrosse with inputs from Martina Ulvrova and Adrien Morison
  Date: 2015/09/11
"""
import numpy as np
import matplotlib.pyplot as plt
import f90nml
import os
import sys
import seaborn as sns

def rprof_cmd(args):
    '''
    Function to plot radial profiles
    '''
    istart, ilast, istep = args.timestep

    ############### INPUT PARAMETERS ################
    # should rather be default parameters to be replaced by command line options

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
    rmin = 1.19
    rmax = rmin + 1.
#    xi0l = 0.1
#    DFe = 0.85
#    xi0s = DFe*xi0l
#    xieut = 0.8

    #Read par file in the parent or present directory. Should be optional or tested
    # should be a separated func in misc module
    read_par_file = True
    if os.path.exists('../par'):
        par_file = '../par'
    elif os.path.exists('par'):
        par_file = 'par'
    else:
        print 'No par file found. Input pars by hand'
        read_par_file = False
        rcmb = 1
        geom = str(raw_input('spherical (s) or cartesian (c)? '))
        spherical = geom == 's'

    if read_par_file:
        nml = f90nml.read(par_file)
        spherical = (nml['geometry']['shape'] == 'spherical' or
                     nml['geometry']['shape'] == 'Spherical')
        if spherical:
            rcmb = nml['geometry']['r_cmb']
        else:
            rcmb = 0.
        stem = nml['ioin']['output_file_stem']
        ste = stem.split('/')[-1]
#        proffile = ste+'_rprof.dat'
        if os.path.exists('../'+ste+'_rprof.dat'):
            proffile = '../'+ste+'_rprof.dat'
        elif os.path.exists(ste+'_rprof.dat'):
            proffile = ste+'_rprof.dat'
        else:
            print 'No profile file found.'
            sys.exit()

        rmin = rcmb
        rmax = rcmb+1
        # if 'fe_eut' in nml['tracersin']:
        #     xieut = nml['tracersin']['fe_eut']
        # if 'k_fe' in nml['tracersin']:
        #     DFe = nml['tracersin']['k_fe']
        # if 'fe_cont' in nml['tracersin']:
        #     xi0l = nml['tracersin']['fe_cont']

#    xi0s = DFe*xi0l
#    xired = xi0l/xieut
#    Rsup = (rmax**3-xired**(1/(1-DFe))*(rmax**3-rmin**3))**(1./3.)

    # def initprof(rpos):
    #     """Theoretical profile at the end of magma ocean crystallization"""
    #     if rpos < Rsup:
    #         return xi0s*((rmax**3-rmin**3)/(rmax**3-rpos**3))**(1-DFe)
    #     else:
    #         return xieut

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
    for iti in range(0, nsteps-1):
        nzp = np.append(nzp, tsteps[iti+1, 0]-tsteps[iti, 0]-1)

    nzp = np.append(nzp, lnum-tsteps[nsteps-1, 0])

    nzs = [[0, 0, 0]]
    nzc = 0
    for iti in range(1, nsteps):
        if nzp[iti] != nzp[iti-1]:
            nzs.append([iti, iti-nzc, int(nzp[iti-1])])
            nzc = iti
    if nzp[nsteps-1] != nzs[-1][1]:
        nzs.append([nsteps, nsteps-nzc, int(nzp[nsteps-1])])

    nzi = np.array(nzs)

    def plotprofiles(quant, *vartuple, **kwargs):
        """Plot the chosen profiles for the chosen timesteps

        quant holds the strings for the x axis annotation and
        the legends for the additional profiles

        vartuple contains the numbers of the column to be plotted

        A kwarg should be used for different options, e.g. whether
        densities of total values are plotted
        """
        if kwargs != {}:
            for key, value in kwargs.iteritems():
                if key == 'integrated':
                    integrate = value
                else:
                    print "kwarg value in plotprofile not understood %s == %s" %(key, value)
                    print "ignored"
        else:
            integrate = False

        if integrate:
            integ = lambda f, r: f * (r/rmax)**2
        if quant[0] == 'Grid':
#            fig, ax = plt.subplots(2, sharex=True)
            fig, axe = plt.subplots(2, sharex=True)
        else:
            fig, axe = plt.subplots()
#            fig, ax = plt.subplots()
        for step in range(istart, ilast, istep):
            step = step +1# starts at 0=> 15 is the 16th
            # find the indices
            ann = sorted(np.append(nzi[:, 0], step))
            inn = ann.index(step)

            nnz = np.multiply(nzi[:, 1], nzi[:, 2])

            i0 = np.sum([nnz[0:inn]])+(step-nzi[inn-1, 0]-1)*nzi[inn, 2]
            i1 = i0+nzi[inn, 2]-1

            if integrate:
                radius = np.array(data[i0:i1, 0], float) + rcmb

            #Plot the profiles
            if quant[0] == 'Grid':
                axe[0].plot(data[i0:i1, 0], '-ko', label='z')
                axe[0].set_ylabel('z', fontsize=ftsz)

                dz = np.array(data[i0+1:i1, 0], np.float) - np.array(data[i0:i1-1, 0], np.float)
                axe[1].plot(dz, '-ko', label='dz')
                axe[1].set_xlabel('cell number', fontsize=ftsz)
                axe[1].set_ylabel('dz', fontsize=ftsz)
            else:
                for i, j in enumerate(vartuple):
                    if i == 0:
                        if integrate:
                            donnee = map(integ, np.array(data[i0:i1, j], float), radius)
                            pplot = plt.plot(donnee, data[i0:i1, 0], linewidth=lwdth,
                                             label=r'$t=%.2e$' % (tsteps[step-1, 2]))
                        else:
                            pplot = plt.plot(data[i0:i1, j], data[i0:i1, 0], linewidth=lwdth,
                                             label=r'$t=%.2e$' % (tsteps[step-1, 2]))
                        col = pplot[0].get_color()
                        lstyle = pplot[0].get_linestyle()
                        axes = plt.gca()
                        rangex = axes.get_xlim()
                        rangey = axes.get_ylim()
                        if ((quant[0] == 'Concentration' or
                             quant[0] == 'Temperature') and
                                plot_conctheo and step == istart+1):
                            rin = np.array(data[i0:i1, 0], np.float)+rmin
                            rfin = (rmax**3.+rmin**3.-rin**3.)**(1./3.)-rmin
                            plt.plot(data[i0:i1, j], rfin, 'b--', linewidth=lwdth,
                                     label='Overturned')
                    else:
                        if i == 1:
                            lstyle = '--'
                        elif i == 2:
                            lstyle = '-.'
                        else:
                            lstyle = ':'
                        if integrate:
                            donnee = map(integ, np.array(data[i0:i1, j], float), radius)
                            plt.plot(donnee, data[i0:i1, 0], c=col,
                                     linestyle=lstyle, linewidth=lwdth)
                        else:
                            plt.plot(data[i0:i1, j], data[i0:i1, 0], c=col,
                                     linestyle=lstyle, linewidth=lwdth)
                    if len(vartuple) > 1:
                        ylgd = rangey[1]-0.05*i*(rangey[1]-rangey[0])
                        xlgd1 = rangex[1]-0.12*(rangex[1]-rangex[0])
                        xlgd2 = rangex[1]-0.05*(rangex[1]-rangex[0])
                        plt.plot([xlgd1, xlgd2], [ylgd, ylgd], c='black',
                                 linestyle=lstyle, linewidth=lwdth)
                        plt.text(xlgd1-0.02*(rangex[1]-rangex[0]), ylgd, quant[i+1], ha='right')

                    plt.ylim([-0.05, 1.05])

                    plt.xlabel(quant[0], fontsize=ftsz)
                    plt.ylabel('z', fontsize=ftsz)
                    plt.xticks(fontsize=ftsz)
                    plt.yticks(fontsize=ftsz)
        if quant[0] == 'Grid':
            plt.savefig("Grid.pdf", format='PDF')
        else:
            lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                             borderaxespad=0., mode="expand",
                             ncol=3, fontsize=ftsz,
                             columnspacing=1.0, labelspacing=0.0,
                             handletextpad=0.0, handlelength=1.5,
                             fancybox=True, shadow=False)

            plt.savefig(quant[0].replace(' ', '_')+".pdf", format='PDF',
                        bbox_extra_artists=(lgd, ), bbox_inches='tight')
        plt.close(fig)
        return

    # Now use it for the different types of profiles

    if plot_temperature:
        if plot_minmaxtemp:
            plotprofiles(['Temperature', 'Mean', 'Minimum', 'Maximum'], 1, 2, 3,
                         integrated=False)
        else:
            plotprofiles(['Temperature'], 1)

    if plot_velocity:
        if plot_minmaxvelo:
            plotprofiles(['Vertical Velocity', 'Mean', 'Minimum', 'Maximum'],
                         7, 8, 9)
        else:
            plotprofiles(['Vertical Velocity'], 7)

    if plot_viscosity:
        if plot_minmaxvisco:
            plotprofiles(['Viscosity', 'Mean', 'Minimum', 'Maximum'], 13, 14, 15)
        else:
            plotprofiles(['Viscosity'], 13)

    if plot_concentration:
        if plot_minmaxcon:
            plotprofiles(['Concentration', 'Mean', 'Minimum', 'Maximum'],
                         36, 37, 38)
        else:
            plotprofiles(['Concentration'], 36)

    # Plot grid spacing
    if plot_grid:
        plotprofiles(['Grid'])

    # Plot the profiles of vertical advection: total and contributions from up-
    # and down-welling currents
    if plot_advection:
        plotprofiles(['Advection per unit surface', 'Total', 'down-welling',
                      'Up-welling'], 57, 58, 59)
        if spherical:
            plotprofiles(['Total scaled advection', 'Total', 'down-welling',
                          'Up-welling'], 57, 58, 59, integrated=True)

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
    #     if spherical:
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
