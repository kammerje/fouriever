from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import klcal
from fouriever import uvfit


# =============================================================================
# MAIN
# =============================================================================

# Kernel phase test data.
scidir = 'data/HIP50156_kpfiles_sci/'
scifiles = [f for f in os.listdir(scidir) if f.endswith('kernel.fits')]
caldir = 'data/HIP50156_kpfiles_cal/'
calfiles = [f for f in os.listdir(caldir) if f.endswith('kernel.fits')]
odir = 'data/HIP50156_kpfiles/'

# Perform Karhunen-Loeve calibration.
data = klcal.data(scidir=scidir,
                  scifiles=scifiles,
                  caldir=caldir,
                  calfiles=calfiles)
data.calibrate(odir=odir,
               K_klip=4) # order of Karhunen-Loeve calibration.

# Kernel phase test data.
idir = 'data/HIP50156_kpfiles/'
fitsfiles = [f for f in os.listdir(idir) if f.endswith('klcal.fits')] # real data of HIP 50156
# idir = 'data/V410Tau_kpfiles/'
# fitsfiles = [f for f in os.listdir(idir) if f.endswith('kpfile.fits')] # real data of V410 Tau

# Perform model fitting.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)
fit = data.gridsearch(model='bin', # fit unresolved companion
                      cov=True, # this data set has covariance
                      sep_range=(40., 400.), # use custom separation range
                      step_size=20., # use custom step size
                      smear=None, # use no bandwidth smearing
                      ofile='figures/HIP50156_kl4') # save figures
                      # ofile='figures/V410Tau_nocov') # save figures
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=True, # this data set has covariance
                smear=None, # use no bandwidth smearing
                ofile='figures/HIP50156_kl4') # save figures
                # ofile='figures/V410Tau_nocov') # save figures
fit_sub = data.gridsearch_sub(fit_sub=fit, # best fit from MCMC
                              model='bin', # fit uniform disk with an unresolved companion
                              cov=True, # this data set has covariance
                              sep_range=(40., 400.), # use custom separation range
                              step_size=20., # use custom step size
                              smear=None, # use no bandwidth smearing
                              ofile='figures/HIP50156_kl4_sub') # save figures
                              # ofile='figures/V410Tau_nocov_sub') # save figures
