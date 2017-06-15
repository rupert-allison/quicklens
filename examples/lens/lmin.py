#/usr/bin/env python
# --
# quicklens/examples/lens/ra_lensing.py
# --
# Created: 29th October 2016
# Based on ./make_lensing_estimators.py
# generates a set of lensed maps in the flat-sky limit, 
# adds to these a fixed dust field (appropriate for 150 GHz obs)
# then runs quadratic lensing estimators on them to estimate phi. 
# plots the auto-spectrum of the phi estimates as well as 
# their cross-spectra with the input phi realization, 
# and a semi-analytical estimate of the noise bias.

import numpy as np
import pylab as pl
from copy import deepcopy

import quicklens as ql
pl.rc('font',**{'family':'serif','serif':['Computer Modern'],'size':16})

## Faint/Bright dust field
field = 'Bfilt'

# simulation parameters.
nsims      = 5
lmin       = 2
lmax       = 3000

ellmin = '%03d' % lmin

nx = 3480 # number of pixels.
ny = 2400 # number of pixels.

dx = 0.499768 / 60. / 180.*np.pi	
dy = 0.502452 / 60. / 180.*np.pi

nlev_t     = 1e-6 / np.sqrt(2)
nlev_p     = 1e-6  # polarization noise level, in uK.arcmin or as above. 
bl         = ql.spec.bl(1., lmax) # beam transfer function.

pix        = ql.maps.pix(nx,dx,ny=ny,dy=dy)

# analysis parameters
estimators = [ ('ptt', 'r'), ('peb', 'b') ] # (estimator, plotting color color) pairs ('ptt' = TT estimator, etc.)

## RA: note the reverse ordering of nx <-> ny here. 
#x, y = np.meshgrid( np.arange(0,nx), np.arange(0,ny) )
#mask = np.sin( np.pi/ny*y )*np.sin( np.pi/nx*x )
mask       = np.ones( (ny, nx) ) # mask to apply when inverse-variance filtering.
                                 # currently, no masking.
                                 # alternatively, cosine masking:
                                 # x, y = np.meshgrid( np.arange(0,nx), np.arange(0,nx) )
                                 # mask = np.sin( np.pi/nx*x )*np.sin( np.pi/nx*y )

mc_sims_mf = None                # indices of simulations to use for estimating a mean-field.
                                 # currently, no mean-field subtraction.
                                 # alternatively: np.arange(nsims, 2*nsims)

npad = 1

# plotting parameters.
t          = lambda l: (l+0.5)**4/(2.*np.pi) # scaling to apply to cl_phiphi when plotting.
lbins      = np.linspace(2, lmax, 50)       # multipole bins.

# cosmology parameters.
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)
clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx,ny=ny,dy=dy)).get_ml(lbins, t=t)   ## RA comment: binned version of the theory lensing power spectrum

# make libraries for simulated skies.
# Add dust power spectra for inverse-variance filtering
if field == 'noDust':
	sky_lib    = ql.sims.cmb.library_flat_lensed_dust(pix, cl_unl, field+"/sky", "/Users/allisonradmin/Documents/Cambridge/core/data/noDust/")
	print 'No dust added' 
if field == 'B':
	sky_lib    = ql.sims.cmb.library_flat_lensed_dust(pix, cl_unl, field+"/sky", "/Users/allisonradmin/Documents/Cambridge/core/data/bright/")
	print 'Bright patch'
if field == 'Bfilt':
	sky_lib    = ql.sims.cmb.library_flat_lensed_dust(pix, cl_unl, field+"/sky", "/Users/allisonradmin/Documents/Cambridge/core/data/bright/")
	print 'Bright patch, including dust power in lensing filters'

obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir=field+"/obs")

cl_lenDust = cl_len.copy()
if field is 'Bfilt':
	alpha	= -0.61
	A	= 114.3 
	dustTTmodel = np.nan_to_num( 2 * np.pi * A * (cl_len.ls / 80.)**(alpha - 2.) / (80.**2) )
	alpha	= -0.88
	A	= 2.5 
	dustEEmodel = np.nan_to_num( 2 * np.pi * A * (cl_len.ls / 80.)**(alpha - 2.) / (80.**2) )
	alpha	= -0.77
	A	= 1.7 
	dustBBmodel = np.nan_to_num( 2 * np.pi * A * (cl_len.ls / 80.)**(alpha - 2.) / (80.**2) )
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_len.cltt / 2 / np.pi, c = 'k') 
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_len.clee / 2 / np.pi, c = 'blue') 
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_len.clbb / 2 / np.pi, c = 'green') 
	cl_lenDust.cltt += dustTTmodel
	cl_lenDust.clee += dustEEmodel
	cl_lenDust.clbb += dustBBmodel 
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_lenDust.cltt / 2 / np.pi, c = '0.7') 
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_lenDust.clee / 2 / np.pi, c = 'yellow') 
	#pl.plot( cl_len.ls, cl_len.ls * (cl_len.ls + 1) * cl_lenDust.clbb / 2 / np.pi, c = 'cyan') 
	#pl.show()

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_lenDust.cltt, 'clee' : cl_lenDust.clee, 'clbb' : cl_lenDust.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir=field+"/qest", npad=npad)

qest_lib_kappa = ql.sims.qest.library_kappa(qest_lib, sky_lib)

qecl_lib = ql.sims.qecl.library(qest_lib, lib_dir=field+"/qecl", mc_sims_mf=mc_sims_mf, npad=npad)
qecl_kappa_cross_lib = ql.sims.qecl.library(qest_lib, qeB=qest_lib_kappa, lib_dir=field+"/qecl_kappa", npad=npad)

# --
# run estimators, make plots.
# --

pl.figure()

p = pl.plot
#p = pl.semilogy

cl_unl.plot('clpp', t=t, color='k', p=p)
clpp.plot( p = pl.errorbar, color = 'k', ls = 'None', marker = 'x', markersize = 3)  

for key, color in estimators:
    qr = qest_lib.get_qr(key)
    qcr = qecl_lib.get_qcr_lcl(key)

    # intialize averagers.
    #cl_phi_x_phi_avg    = ql.util.avg()
    #cl_phi_x_est_avg    = ql.util.avg()
    cl_est_x_est_avg    = ql.util.avg()
    cl_est_x_est_n0_avg = ql.util.avg()
    cl_secondMoment_avg = ql.util.avg()
    #cl_debiased_avg = ql.util.avg()

    # average power spectrum estimates.
    for idx, i in ql.util.enumerate_progress(np.arange(0, nsims), label="averaging cls for key=%s"%key):
        n0 = (qecl_lib.get_sim_ncl_lcl(key, i) / qcr).get_ml(lbins, t=t)
	ps = (qecl_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) 
	cl_est_x_est_n0_avg.add( n0 )
        cl_est_x_est_avg.add( ps )
        cl_secondMoment_avg.add( ps*ps - ps*n0 - ps*n0 + n0*n0 )
        #cl_phi_x_est_avg.add( (qecl_kappa_cross_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
        #cl_debiased_avg.add( (qecl_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) - (qecl_lib.get_sim_ncl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
   
    # handle N1 subtraction (or no subtraction, if you like) 
    doN1sub = True
    ls    = cl_est_x_est_avg.ls
    lbins = cl_est_x_est_avg.lbins
    n1 = np.zeros_like( ls )
    if doN1sub:
        # Load in N1 curves and interpolate onto the bin positions ls
	# File type should be as output by AL's Fortran N1 code.
        n1 = np.loadtxt('/Users/allisonradmin/Documents/Cambridge/core/data/N1_All_rupert.dat')
	
	n1_L = n1[:, 0]
	if key is 'ptt':
           #n00 = n00[:, 2]
           n1 = n1[:, 1]
	if key is 'peb':
           #n00 = n00[:, 16]
           n1 = n1[:, 15]
        n1 = np.interp( ls, n1_L, n1 * t(n1_L) )
    n1 = ql.spec.bcl( lbins, {'cl': n1 } )
    n1.plot(p=p, color = color, lw = 1, ls = ':' ) 
    
    # plot lensing spectra.
    n0 = (1./qr).get_ml(lbins, t=t)
    #(n0).plot(color='y')            # analytical n0 bias.
    #cl_phi_x_est_avg.plot(p=p, color=color)               # lensing estimate x input lensing. 
    #cl_est_x_est_avg.plot(p=p, color=color, lw=3)         # lensing estimate auto spectrum.
    cl_est_x_est_n0_avg.plot(p=p, color=color, ls='--' )   # semi-analytical n0 bias.
    fsky = nx*ny*dx*dy / ( 4 * np.pi )
    dL = lbins[1:] - lbins[:-1]
    yerr   = np.sqrt( 2. / (2 * ls + 1) / fsky / dL / nsims) * cl_est_x_est_avg.specs['cl'] ## Standard error on the mean sim
    #yerrMC = np.sqrt( ( nsims / (nsims - 1.) ) * ( cl_secondMoment_avg.specs['cl'] - (cl_est_x_est_avg - cl_est_x_est_n0_avg).specs['cl']**2 ) / nsims ) 
    (cl_est_x_est_avg - cl_est_x_est_n0_avg - n1).plot(p=pl.errorbar, color=color, lw=1, ls = '-', yerr = yerr, marker = 'o', markersize = 2)         # Debiased lensing auto spectrum.
    np.savetxt( '/Users/allisonradmin/Documents/Cambridge/core/data/lmin_'+ellmin+'_clpp_'+key+'_'+field+'_noNoise_Bfilt.dat', np.real(np.vstack( ( ls, (cl_est_x_est_avg - cl_est_x_est_n0_avg - n1).specs['cl'], clpp.specs['cl'], yerr) ) ).T, fmt = '%.5e' )

pl.xlabel(r'$l$')
pl.ylabel(r'$(l+\frac{1}{2})^4 C_l^{\phi\phi} / 2\pi$')

pl.xlim(0, lbins[-1])
pl.ylim(-0.2e-7, 1.8e-7)

pl.hlines( 0., 0., lmax, linestyles = ':', colors = 'k' )
for key, color in estimators:
    pl.plot( [-1,-2], [-1,-2], label=r'$\phi^{' + key[1:].upper() + r'}$', lw=2, color=color )
pl.legend(loc='upper right')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)
    
#pl.ion()
pl.tight_layout()
pl.show()
