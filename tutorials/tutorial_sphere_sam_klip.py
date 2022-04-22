from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import intercorr, klcal, uvfit


# =============================================================================
# MAIN
# =============================================================================

# SPHERE/SAM test data.
sci_idir = '/Users/jkammerer/Downloads/hd142527_sci/'
sci_odir = 'data/HD142527_uncal_cov/'
cal_idir = '/Users/jkammerer/Downloads/hd142695_cal/'
cal_odir = 'data/HD142695_uncal_cov/'
odir = 'data/HD142695_cov/'
fdir = 'figures_sphere_sam/'

# Load data.
data = intercorr.data(idir=sci_idir,
                      fitsfiles=None)

# Add covariance.
data.clear_cov()
data.add_t3cov(odir=sci_odir)

# Load data.
data = intercorr.data(idir=cal_idir,
                      fitsfiles=None)

# Add covariance.
data.clear_cov()
data.add_t3cov(odir=cal_odir)

# Load data.
data = klcal.data(scidir=sci_odir,
                  scifiles=None,
                  caldir=cal_odir,
                  calfiles=None)

# Perform Karhunen-Loeve calibration.
data.calibrate(odir=odir,
               K_klip=10) # order of Karhunen-Loeve calibration.

# Load data.
fitsfiles = [f for f in os.listdir(odir) if f.endswith('.oifits')] # real data of HD 142527
data = uvfit.data(idir=odir,
                  fitsfiles=fitsfiles)
data.set_observables(['t3'])

# Compute chi-squared map.
fit = data.chi2map(model='bin', # fit unresolved companion
                   cov=True, # this data set has covariance
                   sep_range=(10., 100.), # use custom separation range
                   step_size=10., # use custom step size
                   smear=None, # use no bandwidth smearing
                   ofile=fdir+'hd142527_cov') # save figures

# Run MCMC around best fit position.
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=True, # this data set has covariance
                smear=None, # use no bandwidth smearing
                ofile=fdir+'hd142527_cov') # save figures

# Compute chi-squared map after subtracting best fit companion.
fit_sub = data.chi2map_sub(fit_sub=fit, # best fit from MCMC
                           model='bin', # fit unresolved companion
                           cov=True, # this data set has covariance
                           sep_range=(10., 100.), # use custom separation range
                           step_size=10., # use custom step size
                           smear=None, # use no bandwidth smearing
                           ofile=fdir+'hd142527_cov_sub') # save figures
