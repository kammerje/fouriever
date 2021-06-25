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

# GRAVITY test data.
idir = 'data/'
fitsfiles = ['betaPic_00deg.oifits'] # simulated data of beta Pic
# fitsfiles = ['betaPic_90deg.oifits'] # simulated data of beta Pic
# fitsfiles = ['GRAVI.2018-04-18T08-08-19.739_singlescivis_singlesciviscalibrated.fits'] # real data of HIP 78183

# Perform model fitting.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)
fit = data.gridsearch(model='ud_bin', # fit uniform disk with an unresolved companion
                      cov=False, # this data set has no covariance
                      sep_range=(4., 40.), # use custom separation range
                      step_size=2., # use custom step size
                      smear=3, # use bandwidth smearing of 3
                      ofile='figures/betaPic_00deg') # save figures
                      # ofile='figures/betaPic_90deg') # save figures
                      # ofile='figures/HIP78183') # save figures
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=False, # this data set has no covariance
                smear=3, # use bandwidth smearing of 3
                ofile='figures/betaPic_00deg') # save figures
                # ofile='figures/betaPic_90deg') # save figures
                # ofile='figures/HIP78183') # save figures
fit_sub = data.gridsearch_sub(fit_sub=fit, # best fit from MCMC
                              model='ud_bin', # fit uniform disk with an unresolved companion
                              cov=False, # this data set has no covariance
                              sep_range=(4., 40.), # use custom separation range
                              step_size=2., # use custom step size
                              smear=3, # use bandwidth smearing of 3
                              ofile='figures/betaPic_00deg_sub') # save figures
                              # ofile='figures/betaPic_90deg_sub') # save figures
                              # ofile='figures/HIP78183_sub') # save figures
