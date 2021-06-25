from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import uvfit


# =============================================================================
# MAIN
# =============================================================================

# NIRISS/AMI test data.
idir = 'data/'
fitsfiles = ['ABDor_NIRISS_F480M.oifits'] # simulated data of AB Dor

# Perform model fitting.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)
fit = data.gridsearch(model='bin', # fit uniform disk with an unresolved companion
                      cov=False, # this data set has no covariance
                      sep_range=(50., 500.), # use custom separation range
                      step_size=20., # use custom step size
                      smear=None, # use no bandwidth smearing
                      ofile='figures/abdor') # save figures
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=False, # this data set has no covariance
                smear=None, # use no bandwidth smearing
                ofile='figures/abdor') # save figures
fit_sub = data.gridsearch_sub(fit_sub=fit, # best fit from MCMC
                              model='bin', # fit uniform disk with an unresolved companion
                              cov=False, # this data set has no covariance
                              sep_range=(50., 500.), # use custom separation range
                              step_size=20., # use custom step size
                              smear=None, # use no bandwidth smearing
                              ofile='figures/abdor_sub') # save figures
