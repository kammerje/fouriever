from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import klcal, uvfit


# =============================================================================
# MAIN
# =============================================================================

# Kernel phase test data.
scidir = 'data/HIP50156_kpfiles_sci/'
scifiles = [f for f in os.listdir(scidir) if f.endswith('kernel.fits')] # real data of HIP 50156
caldir = 'data/HIP50156_kpfiles_cal/'
calfiles = [f for f in os.listdir(caldir) if f.endswith('kernel.fits')] # real data of calibrators for HIP 50156
odir = 'data/HIP50156_kpfiles/'

# Load data.
data = klcal.data(scidir=scidir,
                  scifiles=scifiles,
                  caldir=caldir,
                  calfiles=calfiles)

# Perform Karhunen-Loeve calibration.
data.calibrate(odir=odir,
               K_klip=50) # order of Karhunen-Loeve calibration.

# Kernel phase test data.
idir = 'data/HIP50156_kpfiles/'
fitsfiles = [f for f in os.listdir(idir) if f.endswith('klcal.fits')] # real data of HIP 50156
# idir = 'data/V410Tau_kpfiles/'
# fitsfiles = [f for f in os.listdir(idir) if f.endswith('kpfile.fits')] # real data of V410 Tau

# Load data.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)

# Compute linear contrast map.
fit = data.lincmap(cov=True, # this data set has covariance
                   sep_range=(40., 400.), # use custom separation range
                   step_size=5., # use custom step size
                   smear=None, # use no bandwidth smearing
                   ofile='figures/HIP50156', # save figures
                   # ofile='figures/V410Tau', # save figures
                   save_as_fits=True) # save fits file

# Compute chi-squared map.
fit = data.chi2map(model='bin', # fit unresolved companion
                   cov=True, # this data set has covariance
                   sep_range=(40., 400.), # use custom separation range
                   step_size=20., # use custom step size
                   smear=None, # use no bandwidth smearing
                   ofile='figures/HIP50156') # save figures
                   # ofile='figures/V410Tau') # save figures

# Run MCMC around best fit position.
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=True, # this data set has covariance
                smear=None, # use no bandwidth smearing
                ofile='figures/HIP50156') # save figures
                # ofile='figures/V410Tau') # save figures

# Compute chi-squared map after subtracting best fit companion.
fit_sub = data.chi2map_sub(fit_sub=fit, # best fit from MCMC
                           model='bin', # fit uniform disk with an unresolved companion
                           cov=True, # this data set has covariance
                           sep_range=(40., 400.), # use custom separation range
                           step_size=20., # use custom step size
                           smear=None, # use no bandwidth smearing
                           ofile='figures/HIP50156_sub') # save figures
                           # ofile='figures/V410Tau_sub') # save figures
