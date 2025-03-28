from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import intercorr, uvfit


# =============================================================================
# MAIN
# =============================================================================

# NIRISS/AMI test data.
idir = '../data/ABDor/'
odir = '../data/ABDor_cov/'
fitsfiles = ['ABDor_NIRISS_F480M.oifits']

# Load data.
data = intercorr.data(idir=idir,
                      fitsfiles=fitsfiles)

# Add covariance.
data.clear_cov()
data.add_cpcov(odir=odir)

# Load data.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)

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
                ofile='../figures/abdor_smear_cov', # save figures
                sampler='emcee') # sampling algorithm

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
            step_size=25., # use custom step size
            smear=3, # use bandwidth smearing of 3
            ofile='../figures/abdor_smear_cov_sub') # save figures
