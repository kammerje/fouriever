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

# PIONIER test data.
idir = 'data/'
fitsfiles = ['AXCir.oifits'] # real data of AX Cir

# Perform model fitting.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)
fit = data.gridsearch(model='ud_bin', # fit uniform disk with an unresolved companion
                      cov=False, # this data set has no covariance
                      sep_range=None, # use default separation range
                      step_size=None, # use default step size
                      smear=3, # use bandwidth smearing of 3
                      ofile='figures/axcir_smear') # save figures
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=False, # this data set has no covariance
                smear=3, # use bandwidth smearing of 3
                ofile='figures/axcir_smear') # save figures
fit_sub = data.gridsearch_sub(fit_sub=fit, # best fit from MCMC
                              model='ud_bin', # fit uniform disk with an unresolved companion
                              cov=False, # this data set has no covariance
                              sep_range=None, # use default separation range
                              step_size=None, # use default step size
                              smear=3, # use bandwidth smearing of 3
                              ofile='figures/axcir_smear_sub') # save figures
