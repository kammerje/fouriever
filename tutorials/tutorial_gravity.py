from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os

from fouriever import uvfit
from scipy.stats import pearsonr

# =============================================================================
# MAIN
# =============================================================================

# GRAVITY test data.
idir = '../data/GRAVITY/'
fitsfiles = ['betaPic_00deg.oifits'] # simulated data of beta Pic
# fitsfiles = ['betaPic_90deg.oifits'] # simulated data of beta Pic
# fitsfiles = ['GRAVI.2018-04-18T08-08-19.739_singlescivis_singlesciviscalibrated.fits'] # real data of HIP 78183

# Load data.
data = uvfit.data(idir=idir,
                  fitsfiles=fitsfiles)

# Compute chi-squared map.
fit = data.chi2map(model='ud_bin', # fit uniform disk with unresolved companion
                   cov=False, # this data set has no covariance
                   sep_range=(4., 40.), # use custom separation range
                   step_size=2., # use custom step size
                   smear=3, # use bandwidth smearing of 3
                   ofile='../figures/betaPic_00deg') # save figures
                   # ofile='../figures/betaPic_90deg') # save figures
                   # ofile='../figures/HIP78183') # save figures

# Run MCMC around best fit position.
fit = data.mcmc(fit=fit, # best fit from gridsearch
                temp=None, # use default temperature (reduced chi-squared of best fit)
                cov=False, # this data set has no covariance
                smear=3, # use bandwidth smearing of 3
                ofile='../figures/betaPic_00deg', # save figures
                # ofile='../figures/betaPic_90deg', # save figures
                # ofile='../figures/HIP78183', # save figures
                sampler='emcee') # sampling algorithm

samples=fit['samples']
ras=samples[:,1]
decs=samples[:,2]
print("median RA offset:",np.median(ras))
print("median DEC offset:",np.median(decs))
print("sigma RA offset:",np.std(ras))
print("sigma DEC offset:",np.std(decs))
print("pearson rho:",pearsonr(ras,decs).statistic)

# Compute chi-squared map after subtracting best fit companion.
fit_sub = data.chi2map_sub(fit_sub=fit, # best fit from MCMC
                           model='ud_bin', # fit uniform disk with unresolved companion
                           cov=False, # this data set has no covariance
                           sep_range=(4., 40.), # use custom separation range
                           step_size=2., # use custom step size
                           smear=3, # use bandwidth smearing of 3
                           ofile='../figures/betaPic_00deg_sub') # save figures
                           # ofile='../figures/betaPic_90deg_sub') # save figures
                           # ofile='../figures/HIP78183_sub') # save figures
