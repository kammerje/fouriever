from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import intercorr, uvfit

import sys
sys.path.append('/Users/jkammerer/Documents/Code/CANDID')
import candid


# =============================================================================
# MAIN
# =============================================================================

# NIRISS/AMI test data.
idir = '../data/ABDor/'
odir = '../data/ABDor_cov/'
fitsfiles = ['ABDor_NIRISS_F480M.oifits'] # simulated data of AB Dor
# fitsfiles = ['obs001_pri1_sub0_calib_obs004_pri1_sub0_F380M.oifits']
# fitsfiles = ['obs001_pri1_sub0_calib_obs004_pri1_sub0_F480M.oifits']

# Load data.
data = intercorr.data(idir=idir,
                      fitsfiles=fitsfiles)

# Add covariance.
data.clear_cov()
data.add_t3cov(odir=odir)

# Load data.
data = uvfit.data(idir=odir,
                  fitsfiles=fitsfiles)

# # Estimate detection limits.
# data.detlim(sigma=3., # confidence level of detection limits
#             cov=False, # this data set has covariance
#             sep_range=(50., 500.), # use custom separation range
#             step_size=50., # use custom step size
#             smear=3, # use bandwidth smearing of 3
#             ofile='TEST') # save figures

# # Comparison with CANDID.
# data = candid.Open(odir+fitsfiles[0])
# data.observables = ['cp', 'v2']
# data.detectionLimit(step=50., fig=9, rmin=50., rmax=500.)
# plt.figure(9)
# plt.savefig('TEST_candid.pdf')
# plt.close()

# import pdb; pdb.set_trace()

# Compute chi-squared map.
fit = data.chi2map(model='bin', # fit unresolved companion
                   cov=True, # this data set has covariance
                   sep_range=(50., 500.), # use custom separation range
                   step_size=25., # use custom step size
                   smear=3, # use bandwidth smearing of 3
                   ofile='../figures/abdor_smear_cov') # save figures

# Run MCMC around best fit position.
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=True, # this data set has covariance
                smear=3, # use bandwidth smearing of 3
                ofile='../figures/abdor_smear_cov') # save figures

# Compute chi-squared map after subtracting best fit companion.
fit_sub = data.chi2map_sub(fit_sub=fit, # best fit from MCMC
                           model='bin', # fit unresolved companion
                           cov=True, # this data set has covariance
                           sep_range=(50., 500.), # use custom separation range
                           step_size=25., # use custom step size
                           smear=3, # use bandwidth smearing of 3
                           ofile='../figures/abdor_smear_cov_sub') # save figures

# Estimate detection limits.
data.detlim(sigma=3., # confidence level of detection limits
            fit_sub=fit, # best fit from MCMC
            cov=True, # this data set has covariance
            sep_range=(50., 500.), # use custom separation range
            step_size=50., # use custom step size
            smear=3, # use bandwidth smearing of 3
            ofile='../figures/abdor_smear_cov_sub') # save figures
