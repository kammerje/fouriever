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

# PIONIER test data.
idir = '../data/AXCir/'
odir = '../data/AXCir_cov/'
fitsfiles = ['axcir.oifits'] # real data of AX Cir

# Load data.
data = intercorr.data(idir=idir,
                      fitsfiles=fitsfiles)

# Add covariance.
data.add_cov(odir=odir)

# Load data.
data = uvfit.data(idir=odir,
                  fitsfiles=fitsfiles)

# Compute chi-squared map.
fit = data.chi2map(model='ud_bin', # fit uniform disk with unresolved companion
                   cov=True, # this data set has covariance
                   sep_range=None, # use default separation range
                   step_size=None, # use default step size
                   smear=3, # use bandwidth smearing of 3
                   ofile='figures/axcir_smear_cov') # save figures

# Run MCMC around best fit position.
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=True, # this data set has covariance
                smear=3, # use bandwidth smearing of 3
                ofile='figures/axcir_smear_cov') # save figures

# Compute chi-squared map after subtracting best fit companion.
fit_sub = data.chi2map_sub(fit_sub=fit, # best fit from MCMC
                           model='ud_bin', # fit uniform disk with unresolved companion
                           cov=True, # this data set has covariance
                           sep_range=None, # use default separation range
                           step_size=None, # use default step size
                           smear=3, # use bandwidth smearing of 3
                           ofile='figures/axcir_smear_cov_sub') # save figures

# Estimate detection limits.
data.detlim(sigma=3., # confidence level of detection limits
            fit_sub=fit, # best fit from MCMC
            cov=True, # this data set has covariance
            sep_range=None, # use default separation range
            step_size=None, # use default step size
            smear=3, # use bandwidth smearing of 3
            ofile='figures/axcir_smear_cov_sub') # save figures
