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

# SPHERE/SAM test data.
idir = '/Users/jkammerer/Downloads/oifits2019/'
odir = 'data/HD142527_cov/'
fdir = 'figures2019/'
fitsfiles = [f for f in os.listdir(idir) if f.endswith('.oifits')] # real data of HD 142527

# Load data.
data = intercorr.data(idir=idir,
                      fitsfiles=fitsfiles)

# Add covariance.
data.clear_cov()
data.add_t3cov(odir=odir)

# Load data.
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
