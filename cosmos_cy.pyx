from __future__ import division
import numpy as np
cimport numpy as np
#### To open pyfits files 
from astropy.io import fits as pyfits
#### Stuff used for using libraries for cosmological parameters
import cosmolopy.distance as cd
#### For exit message
import sys
#### For creating/deleting files
import os, errno
#### Check time of computation
import time
#### Plotting
import matplotlib.pyplot as plt
from pylab import pcolor, show, colorbar, xticks, yticks
#%matplotlib inline
#### Smoothing
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import savgol_filter
#### Integration
from scipy.integrate import quad
#### Laplacian
#from scipy.ndimage.filters import laplace
#### Print options
np.set_printoptions(threshold=np.nan)
#### Self designed functions
from gauss_smooth import gauss_smooth_func as gsmooth #Self-designed Gaussian Smoothing function
from hole_fill import low_density_fill #fill locations with less density with appropriate (block-averaged) values
from galaxy_bias import bias_Amara     #galaxy-dark matter bias


# -------------------------------- Header/ metadata --------------------------------------------------------
__author__ = "Siddharth Satpathy"
__date__ = "21 August 2018"
__version__ = "5.0.0"
__email__ = "siddharthsatpathy.ss@gmail.com"
__status__ = "Complete"


# -------------------- Call C function for computation of potential ----------------------------------------
cdef extern from "./potential.h" nogil:
    void pot_field(double* Kap, double* pott, double* Dec_bin_mid, double* RA_bin_mid, double softening, int Dec_len, int RA_len)


# -------------------- Create output directories -----------------------------------------------------------
try:
    os.makedirs('TxtOutput/')
    os.makedirs('Plots/')    
except OSError as e:
    if e.errno != errno.EEXIST:
        raise 


# -------------------------------- Main program ------------------------------------------------------------
def cosmo_cythonize():
    start_time = time.time()


    #----------------------------- Define (some) variables as per C standard -----------------------------------
    cdef int i, j, k, bin_num_RA, bin_num_Dec
    cdef double D_Com_source


    #----------------------------- Folder paths ----------------------------------------------------------------
    fpath = 'COSMOS/'
    TxtDir = 'TxtOutput/'
    PlotDir = 'Plots/'


    #----------------------------- Read Galaxy Data ------------------------------------------------------------
    input_file = fpath + 'COSMOS2015_Laigle_v1.1.fits'
    #pyfits.info(input_file)
    #header_table = pyfits.getheader(input_file,1)
    #print header_table.keys()
    #head = pyfits.getheader(input_file)
    #head.keys()

    File = pyfits.open(input_file)
    #print File.info
    #print File[1].header.keys 

    tbdata = File[1].data


    #----------------------------- Get cosmological parameters -------------------------------------------------
    angg = 'deg' #Choose angg ='rad' to have RA, Dec in radians; choose angg ='deg' to have RA, Dec in degrees. 

    if angg=='deg':
        RA = tbdata['ALPHA_J2000']              #RA in degrees
        Dec = tbdata['DELTA_J2000']             #Dec in degrees

    if angg=='rad':
        RA = tbdata['ALPHA_J2000']*(np.pi/180)  #RA in radians
        Dec = tbdata['DELTA_J2000']*(np.pi/180) #Dec in radians

    flag = tbdata['FLAG_COSMOS']                #FLAG_COSMOS = 1 or 0. FLAG_COSMOS = 1: 2deg2 COSMOS area. 
    obj_type = tbdata['TYPE']                   #TYPE = 0 for galaxy; TYPE = 1 for star; TYPE = 2 for XRay source; TYPE = -9 for failure in fit.
    photoz = tbdata['ZPDF']                     #ZPDF = 0 for star; ZPDF = 9.99 for XRay source; ZPDF = -99 for masked area.

    scale_fac = 1.0/(1.0+photoz)                #Scale factor

    mark_gal = np.where( (obj_type == 0) & (photoz != -99.0) & (flag == 1) )[0]  #Choose galaxies outside of masked area

    RA_gal_init = RA[mark_gal]                  #RA of galaxies outside of masked area
    Dec_gal_init = Dec[mark_gal]                #Dec of galaxies outside of masked area
    photoz_gal_init = photoz[mark_gal]          #Photometric redshift of galaxies outside of masked area


    ''' The following exercise (saving and importing of essential arrays) is done to ensure better accuracy (higher decimal) 
    of numbers in arrays, which is essential for computational accuracy. '''
    lst = ['RA_gal', 'Dec_gal', 'photoz_gal']

    np.savetxt( TxtDir+'RA_gal_init.txt', RA_gal_init )
    np.savetxt( TxtDir+'Dec_gal_init.txt', Dec_gal_init )    
    np.savetxt( TxtDir+'photoz_gal_init.txt', photoz_gal_init)

    RA_gal_init = np.loadtxt( TxtDir+'RA_gal_init.txt' )
    Dec_gal_init = np.loadtxt( TxtDir+'Dec_gal_init.txt' )
    photoz_gal_init = np.loadtxt( TxtDir+'photoz_gal_init.txt' )

    for i in range(len(lst)):
        os.remove( TxtDir+lst[i]+'_init.txt')  #Delete temporarily created files after their use is over.    


    ''' RA range (max-min) is slighly higher than Dec range (max-min). This is a problem since we have 128 cells in both RA and Dec,
    and we want cell size to be same in both RA and Dec. So we throw of some galaxies beyond a certain RA cutoff, such that in the ensuing
    dataset cell dimension of both RA and Dec are same.'''
    RA_maxcut = np.min(RA_gal_init) + ( np.max(Dec_gal_init)-np.min(Dec_gal_init) ) + 0.2/np.power(10,6) 
    mark_equal_ang_binsize = np.where( RA_gal_init<RA_maxcut )[0] 

    RA_gal = RA_gal_init[mark_equal_ang_binsize]
    Dec_gal = Dec_gal_init[mark_equal_ang_binsize]
    photoz_gal = photoz_gal_init[mark_equal_ang_binsize]


    ''' The following exercise (saving and importing of essential arrays) is done to ensure better accuracy (higher decimal) 
    of numbers in arrays, which is essential for computational accuracy. '''
    np.savetxt( TxtDir+'RA_gal_arr.txt', RA_gal )
    np.savetxt( TxtDir+'Dec_gal_arr.txt', Dec_gal )
    np.savetxt( TxtDir+'photoz_gal_arr.txt', photoz_gal)

    RA_gal = np.loadtxt( TxtDir+'RA_gal_arr.txt' )
    Dec_gal = np.loadtxt( TxtDir+'Dec_gal_arr.txt' )  
    photoz_gal = np.loadtxt( TxtDir+'photoz_gal_arr.txt' ) 

    for i in range(len(lst)):
        os.remove( TxtDir+lst[i]+'_arr.txt')  #Delete temporarily created files after their use is over.


    ''' We want the span of RA and Dec to be the same. At a later stage in this code, we will see that both RA and Dec will be divided
    into equal number of bins. And, we want the bin sizes of RA and Dec to be the same.'''
    print 'max-min span of RA is: ', np.max(RA_gal)-np.min(RA_gal) 
    print 'max-min span of Dec is: ', np.max(Dec_gal)-np.min(Dec_gal)

    #----------------------------- Choose parameters -----------------------------------------------------------
    cosmo = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7, 'omega_k_0': 0.0, 'h':0.7, 'H_0' : 70}   #Cosmology (L'Aigle++)

    delta_photoz = 0.1                         #redshift bin size
    bin_photoz = np.arange( np.min(photoz_gal), np.max(photoz_gal), delta_photoz ) #Redshift bins edges
    #bin_num = int(np.ceil( (np.max(photoz_gal) - np.min(photoz_gal))/delta_photoz ))
    #print bin_num
    redshift_source = 2.3                      #2.5807833333 -- Mean spectroscopic redshift of Clamato backlights
    gal_num = mark_gal.shape[0]                #Number of galaxies inside masked area


    #----------------------------- Histogram (Redshift bins) ---------------------------------------------------
    rho_z, photoz_bin_edges = np.histogram( photoz_gal, bins=bin_photoz ) #, range=(np.min(photoz_gal), np.max(photoz_gal)) )

    # photoz_bin_edges is same thing as bin_photoz
    #rho_z = rho_z/gal_num
    #print len(rho_z), np.min(rho_z), np.max(rho_z), np.mean(rho_z)
    photoz_bin_width = photoz_bin_edges[1]-photoz_bin_edges[0]
    photoz_ch = photoz_bin_edges[:-1] + (photoz_bin_width/2)  #midpoints of photoz_bin_edges (photoz_bin_edges=bin_photoz)
    #photoz_ch = np.arange( np.min(photoz_gal), np.max(photoz_gal), delta_photoz ) #This is old. Don't use.


    D_Com_LOS_ch = (cosmo['H_0']/100) * cd.comoving_distance(photoz_ch, **cosmo)   #LoS comoving distances of midpoints of redshift bins
    D_Com_source = (cosmo['H_0']/100) * cd.comoving_distance(redshift_source, **cosmo) #LoS comoving distances of redshift source plane
    scale_factor_ch = 1.0/(1.0+photoz_ch)      #Scale factors corresponding to midpoints of redshift bins


    #----------------------------- Smoothing (Density - Redshift bins) -----------------------------------------

    ''' 1.> Nadaraya-Watson (NW) kernel regression: 
    1a.> http://www.statsmodels.org/devel/generated/
           statsmodels.nonparametric.kernel_regression.KernelReg.html#statsmodels.nonparametric.kernel_regression.KernelReg 

    1b.> https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way       '''
    kr = KernelReg(rho_z, photoz_ch, 'c')
    rhoz_NWpred, rhoz_NWstd = kr.fit(photoz_ch)

    ''' 2.> Gaussian Filter
    2a.> https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.gaussian_filter.html '''
    rhoz_gauss = gaussian_filter1d(rho_z, sigma=1)

    ''' 3.> Savitzky-Golay
    3a.> https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way 
    3b.> https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.signal.savgol_filter.html '''

    rhoz_savitzky = savgol_filter(rho_z, 51, 3) # window size 51, polynomial order 3

    '''4.> Gaussian Smoothing (inefficient hand-implemented method)
    4a.> https://matthew-brett.github.io/teaching/smoothing_intro.html '''

    rhoz_gsmooth = gsmooth(x_vals=photoz_ch, y_vals=rho_z, sigmaa=0.1)


    """ Plotting parameters """
    AxisTextSize = 16
    AxisTickSize = 18
    LabelSz = 14
    mk_size = 2.5

    min_clim = -3*(10**(-8))
    max_clim = 10.3*(10**(-8))
    mk_scale = 0.8


    #----------------------------- Plot 'numbers of galaxies' vs 'redshift' ------------------------------------
    plt.figure(1)
    plt.plot(photoz_ch, rho_z/gal_num, marker='o', ls=' ', markersize=mk_size, color='r', zorder='10', label='COSMOS data')
    #plt.plot(photoz_ch, rhoz_NWpred, marker='o', ls=' ', markersize=mk_size, color='g', zorder='9', label='nonparametric')
    plt.plot(photoz_ch, rhoz_gauss/gal_num, marker='o', ls=' ', markersize=mk_size, color='g', zorder='111', label=r'${\rm Gaussian \ filter} \ (\sigma = 1)$')
    #plt.plot(photoz_ch, rhoz_savitzky, marker='o', ls=' ', markersize=mk_size, color='k', zorder='12', label='savitzky_golay')
    plt.plot(photoz_ch, rhoz_gsmooth/gal_num, marker='o', ls=' ', markersize=mk_size, color='b', zorder='111', label=r'${\rm Gaussian \ smooth \ (Sidd)} \ (\sigma = 0.1)$')

    plt.axvline(x=redshift_source, ls='--', color='k')
    plt.annotate(r'${\rm Mean \ Clamato \ pixel}$', xy=(redshift_source, 0.026), xytext=(3.0, 0.031), fontsize=12\
                    , arrowprops=dict(facecolor='black', shrink=0.01, headwidth = 10, width = 0.75))
    plt.annotate(r'${\rm redshift} \ (z \simeq 2.3)$', xy=(redshift_source+0.3, 0.023), xytext=(3.3, 0.028), fontsize=12)
    plt.legend()

    plt.tick_params(labelsize=AxisTickSize)
    plt.xlabel(r'${\rm Redshift} \ (z)$', fontsize=AxisTextSize+4)
    plt.ylabel(r'${\rm Density} \ (\rho)$', fontsize=AxisTextSize+4)

    plt.xlim([-0.2,6.1])
    #plt.ylim([0,0.045])

    plt.tight_layout()
    plt.savefig(PlotDir+'COSMOS_DensityRedshift.png')

    plt.close()
    #plt.show()


    rhoz_smooth = rhoz_gsmooth  #rhoz_gsmooth or rho_z                #Choose which smoothed rhoz to use


    #----------------------------- Histogram2D (RA,Dec bins) ---------------------------------------------------
    integrand_part1 = D_Com_LOS_ch * (D_Com_source - D_Com_LOS_ch) 
    integrand_part2 = scale_factor_ch * D_Com_source
    integrand_full = np.divide(integrand_part1, integrand_part2 )

    bin_num_RA = 128                            #Number of RA bins in Kappa map
    bin_num_Dec = 128                           #Number of Dec bins in Kappa map
    RA_bin = np.linspace( np.min(RA_gal), np.max(RA_gal), num=bin_num_RA+1 )       #RA Bins in Kappa map
    Dec_bin = np.linspace( np.min(Dec_gal), np.max(Dec_gal), num=bin_num_Dec+1 )   #Dec Bins in Kappa map

    ''' Compute number of galaxies in 3D bins in 3D space with edges = [photoz, RA, Dec]. 
        The calculation of galaxies in bins goes till the limit of available photoz - [0.0025, 5.9025] '''
    overdensity = np.zeros( len(photoz_bin_edges)-1 )   
    print " " 

    for i in range( len(bin_photoz)-1 ):
        mark_bin = np.where( (photoz_gal>=bin_photoz[i]) & (photoz_gal<bin_photoz[i+1]) )[0]
        RA_mark = RA_gal[mark_bin]
        Dec_mark = Dec_gal[mark_bin]
        rho_RADec, RAedges, Decedges = np.histogram2d(RA_mark, Dec_mark, bins=[RA_bin, Dec_bin])
        #rho_RADec = gaussian_filter(rho_RADec, sigma=0.1)   #This was just a test. Don't use this.

        overdensity_g_dz = ( rho_RADec * (bin_num_RA*bin_num_Dec/rhoz_smooth[i]) ) - 1.0   #galaxy overdensity

        ''' Galaxy bias: delta_g/delta_m '''
        bias = bias_Amara(z=photoz_ch[i], bias_type='nn', zcutoff=2.0) 

        overdensity_m_dz = overdensity_g_dz/bias      #dark matter overdensity

        #print i, " ", rho_z[i], " ", mark_bin.shape[0], "  ", np.sum(rho_RADec)
        #overdensity[i] = np.mean(overdensity_m_dz)

        globals()['integrand'+str(i).zfill(4)] = integrand_full[i] * overdensity_m_dz

        #overdensity[i] = np.mean( globals()['integrand'+str(i).zfill(4)] )  


    ''' Check if overdensity rises or falls with redshift. 
        For a correct implementaion of the code, overdensity should not vary too much with redshift. '''
    plt.figure(555)
    plt.plot( np.arange(24), overdensity[0:24], marker='o', ls=' ', markersize=mk_size )
    #plt.ylim([-20,20])
    #plt.xlim([-0.2,24])
    plt.tight_layout()
    plt.close()
    #plt.show()  

    mark_source = np.where( photoz_bin_edges<redshift_source )[0]     #find indices where photoz_bin_edges lies before the backlight source plane 
    photoz_lim_binedge = photoz_bin_edges[mark_source]                #array with photoz_bin_edges which lie before the backlight source plane 

    if ( abs(photoz_bin_edges[len(photoz_lim_binedge)] - redshift_source) < photoz_bin_width/2 ):
        photoz_lim_binedge = np.append( photoz_lim_binedge, photoz_lim_binedge[-1]+photoz_bin_width )

    photoz_lim = photoz_lim_binedge[:-1] + (photoz_bin_width/2)       #midpoints of photoz_lim_binedge          

    D_Com_LOS_lim = (cosmo['H_0']/100) * cd.comoving_distance(photoz_lim, **cosmo)


    #----------------------------- Convergence(Kappa) ----------------------------------------------------------
    cdef np.ndarray[np.double_t, ndim=2] Kappa = np.zeros( [bin_num_RA , bin_num_Dec], dtype=np.float64)   
    cdef np.ndarray[np.double_t, ndim=2] Kappa_holefill = np.zeros( [bin_num_RA , bin_num_Dec], dtype=np.float64)      

    for i in range(0,bin_num_RA):
        for j in range(0,bin_num_Dec):
            arr = np.empty( len(photoz_lim) )

            for k in range( len(photoz_lim) ):
                arr[k] = globals()['integrand'+str(k).zfill(4)][i,j]

            Kappa[i,j] = np.trapz(arr, D_Com_LOS_lim)   


    c = 2.99792458*np.power(10,5)  #in km/s
    Kappa = (3.0/2.0) * ( (cosmo['H_0']*cosmo['H_0']*cosmo['omega_M_0']) / (c**2) ) * Kappa

    print("--- Computation of Kappa finished in %s seconds o_o ---" % (time.time() - start_time))
    print " "

    np.savetxt( TxtDir+'Kappa.txt', Kappa )
    Kappa = np.loadtxt( TxtDir+'Kappa.txt' )

    print "[mean, min, max] of Kappa BEFORE holes are filled: ", np.mean(Kappa), np.min(Kappa), np.max(Kappa)


    #----------------------------- Plot Convergence Field (Kappa) ----------------------------------------------
    plotlim = 'n'

    if plotlim=='y':
        min_clim = -1.3579/np.power(10,7)
        max_clim = -1.3575/np.power(10,7)

    mark_zlim = np.where( photoz_gal<redshift_source )[0]  #find indices where photoz_gal lies before the backlight source plane.

    ''' Linear transformation between array indices and RA (Dec) values. '''
    labmin_RA = 0
    labmax_RA = bin_num_RA
    min_RA = np.min( (RA_gal[mark_zlim]) )
    max_RA = np.max( (RA_gal[mark_zlim]) ) 
    slope_RA = (labmax_RA - labmin_RA)/(max_RA - min_RA)
    intercept_RA = labmin_RA - slope_RA*min_RA

    tick_lbls_RA = np.array([149.5, 150, 150.5])
    tick_locs_RA = np.around( slope_RA*tick_lbls_RA + intercept_RA, decimals=1 )

    labmin_Dec = 0
    labmax_Dec = bin_num_Dec
    min_Dec = np.min( (Dec_gal[mark_zlim]) ) 
    max_Dec = np.max( (Dec_gal[mark_zlim]) )
    slope_Dec = (labmax_Dec - labmin_Dec)/(max_Dec - min_Dec)
    intercept_Dec = labmin_Dec - slope_Dec*min_Dec

    tick_lbls_Dec = np.array([1.5, 2.0, 2.5])
    tick_locs_Dec = np.around( slope_Dec*tick_lbls_Dec + intercept_Dec, decimals=4 )


    ''' Plot the 2D Convergence (Kappa) matrix. '''
    plt.figure(2)
    pcolor(np.transpose(Kappa))
    plt.colorbar()

    if plotlim=='y':
        plt.clim( min_clim , max_clim )

    #plt.xlim([0,128])
    #plt.ylim([0,128])

    plt.xticks(tick_locs_RA, tick_lbls_RA)
    plt.yticks(tick_locs_Dec, tick_lbls_Dec)

    plt.tick_params(labelsize=AxisTickSize)
    plt.xlabel(r'${\rm RA \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.ylabel(r'${\rm Dec \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.title(r'${\rm Convergence,} \ \kappa$', fontsize=AxisTextSize+4)

    plt.tight_layout()
    plt.savefig(PlotDir + 'COSMOS_Convergence.png')
    plt.close()
    #plt.show()


    ''' Test what happens when you do a Gaussian smoothing of Kappa map. '''
    if(0):
        Kappa_gauss = gaussian_filter(Kappa, sigma=1.0)

        plt.figure(3)
        pcolor(np.transpose(Kappa_gauss))
        plt.colorbar()

        if plotlim=='y':
            plt.clim( min_clim , max_clim )

        #plt.xlim([0,128])
        #plt.ylim([0,128])

        plt.xticks(tick_locs_RA, tick_lbls_RA)
        plt.yticks(tick_locs_Dec, tick_lbls_Dec)

        plt.tick_params(labelsize=AxisTickSize)
        plt.xlabel(r'${\rm RA \ (in \ degrees)}$', fontsize=AxisTextSize+4)
        plt.ylabel(r'${\rm Dec \ (in \ degrees)}$', fontsize=AxisTextSize+4)
        plt.title(r'${\rm Convergence,} \ \kappa$', fontsize=AxisTextSize+4)

        plt.tight_layout()
        #plt.savefig( PlotDir + 'COSMOS_Convergence_GaussianSmoothed.png')
        plt.close()
        #plt.show()


    #----------------- Fill regions in the convergence field where there is unexpected low density (holes) -----
    hole_threshold = -0.022 #(thresh = -0.22, when nn & zcutoff = 2.0), #(thresh = -0.22, when nn & zcutoff = 1.5),  #(thresh = -0.032, when gs & zcutoff = 1.0)
    mark_hole = np.where(Kappa<hole_threshold)

    mark_hole_x = (mark_hole[0])  
    mark_hole_y = (mark_hole[1])  

    Kappa_holefill = Kappa.copy()

    neighb = 10
    for i in range(len(mark_hole_x)):
        hole_val = low_density_fill( indx_x=mark_hole_x[i], indx_y=mark_hole_y[i], neighb=neighb, matrx=Kappa_holefill, \
                                     edge=0, bin_num_RA=bin_num_RA, bin_num_Dec=bin_num_Dec )
        Kappa_holefill[ mark_hole_x[i], mark_hole_y[i] ] = hole_val #np.mean(Kappa)


    print "[mean, min, max] of Kappa AFTER holes are filled: ", np.mean(Kappa_holefill), np.min(Kappa_holefill), np.max(Kappa_holefill)

    Kappa_holefill = Kappa_holefill -np.mean(Kappa_holefill)

    print "[mean, min, max] of Kappa AFTER holes are filled, and AFTER subtraction of mean: ", \
            np.mean(Kappa_holefill), np.min(Kappa_holefill), np.max(Kappa_holefill)

    print " "        

    np.savetxt( TxtDir+'Kappa_holefill.txt', Kappa_holefill )
    Kappa_holefill = np.loadtxt( TxtDir+'Kappa_holefill.txt' )


    ''' Plot convergence field after low density regions (holes) have been filled '''
    plt.figure(4)
    pcolor(np.transpose(Kappa_holefill))
    plt.colorbar()

    if plotlim=='y':
        plt.clim( min_clim , max_clim )

    #plt.xlim([0,128])
    #plt.ylim([0,128])

    plt.xticks(tick_locs_RA, tick_lbls_RA)
    plt.yticks(tick_locs_Dec, tick_lbls_Dec)

    plt.tick_params(labelsize=AxisTickSize)
    plt.xlabel(r'${\rm RA \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.ylabel(r'${\rm Dec \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.title(r'${\rm Convergence,} \ \kappa$', fontsize=AxisTextSize+4)

    plt.tight_layout()
    plt.savefig(PlotDir + 'COSMOS_Convergence_holesfilled.png')
    plt.close()
    #plt.show() 


    #----------------------------- Lensing potential (phi) -----------------------------------------------------
    cdef double Dec_cell, RA_cell, Dec_iter, RA_iter, r, softening
    cdef np.ndarray[np.double_t, ndim=1] Dec_bin_mid = np.zeros( (Kappa.shape[0],), dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=1] RA_bin_mid = np.zeros( (Kappa.shape[1],), dtype=np.float64)
    cdef np.ndarray[np.double_t, ndim=2] phi = np.zeros( [Kappa.shape[0], Kappa.shape[1]], dtype=np.float64)

    Dec_bin_mid = (Dec_bin[1:] + Dec_bin[:-1]) / 2     #midpoints of Dec_bin
    RA_bin_mid = (RA_bin[1:] + RA_bin[:-1]) / 2        #midpoints of RA_bin

    softening = 0

    t1_phi = time.time()

    pot_field( &Kappa_holefill[0,0], &phi[0,0], &Dec_bin_mid[0], &RA_bin_mid[0], softening, Kappa.shape[1], Kappa.shape[0] )

    np.savetxt( TxtDir+'potential_COSMOS.txt', phi )
    t2_phi = time.time()    

    print("--- Computation of phi finished in %s seconds o_o ---" % (t2_phi - t1_phi))  

    phi = np.loadtxt( TxtDir+'potential_COSMOS.txt' )


    ''' Plot 2D Convergence potential (phi) '''    
    plt.figure(5)
    pcolor(np.transpose(phi))
    plt.colorbar()

    if plotlim=='y':
        plt.clim( min_clim , max_clim )

    #plt.xlim([0,128])
    #plt.ylim([0,128])

    plt.xticks(tick_locs_RA, tick_lbls_RA)
    plt.yticks(tick_locs_Dec, tick_lbls_Dec)

    plt.tick_params(labelsize=AxisTickSize)
    plt.xlabel(r'${\rm RA \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.ylabel(r'${\rm Dec \ (in \ degrees)}$', fontsize=AxisTextSize+4)
    plt.title(r'${\rm Potential,} \ \phi$', fontsize=AxisTextSize+4)

    plt.tight_layout()
    plt.savefig(PlotDir + 'COSMOS_Potential.png')
    plt.close()
    #plt.show()  
