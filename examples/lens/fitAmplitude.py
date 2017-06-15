## Fit spectrum amplitude from output of lmin.py

import numpy as np
import matplotlib.pyplot as plt

## Construct (diagonal) inverse noise covariance matrix from standard deviation 
## [Optional argument n for length of data if uncertainties are homoskedastic.]
def invNfromSigmas( sig, n = 2 ):
	if isinstance( sig, (int, float) ):
		sig  = sig * np.ones( n ) 
	return np.diag( 1./sig**2 ) 

## Number of sims used to estimate the standard deviation
#Nsims = 100

## Root directory of files of interest:
fileDir = '/Users/allisonradmin/Documents/Cambridge/core/data/'

for key in ['tt', 'eb' ]:
 	print '\nEstimator: ', key
	## Data, modle template and uncertainties
	ell, data, m, sigma = np.loadtxt( fileDir + 'lmin_002_clpp_p'+key+'_Bfilt_noNoise.dat', unpack = True )
	
	## Restrict fit range
	lmax = 2000
	lmin = 2 
	inds = np.where( np.logical_and( ell >= lmin, ell < lmax ) )
	ell = ell[ inds ]
	data = data[ inds ] 
	m    = m[ inds ] 
	sigma = sigma[ inds ]

	## This is overkill, but useful if eventually move to non-diagonal noise covariance.
	Ninv = invNfromSigmas( sigma )

	mTinvN = np.dot( m, Ninv )
	sigA   = 1./np.sqrt( np.dot( mTinvN, m ) ) 
	Abar   = np.dot( mTinvN, data ) * sigA**2

	SNR    = Abar / sigA

	"""
	SNRell = (m / sigma)**2
	plt.plot( ell, SNRell )
	plt.show()
	print np.sqrt( np.sum(SNRell) )
	"""

	chi2   = np.dot( data, np.dot( Ninv, data ) ) + np.dot( Abar * mTinvN, Abar * m - 2 * data )

	print 'Abar   = %.3f' % Abar
	print 'sigA   = %.4f' % sigA
	print 'SNR    = %.1f' % SNR
	print 'chi2bf = %.1f' % chi2 
	print 'dof    = %d' % len(m) 
	print 'sigA (scaled)   = %.3f' % (sigA * 0.32)

	plt.plot( ell, data )
	plt.plot( ell, m, c = 'k' )
	plt.plot( ell, m * Abar, c = 'k' )
plt.show()
