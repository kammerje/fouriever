# fouriever

Calibrations, correlations, and companion search for kernel-phase, aperture masking, and long-baseline interferometry data.

## Description

This toolkit combines different pieces of code that I developed mainly during my PhD. Collaborators include **Mike Ireland**, **Antoine Merand**, and **Frantz Martinache**. The goal is to have a common framework to analyze and fit kernel-phase, aperture masking, and long-baseline interferometry data.

Currently, the toolkit contains the following functionalities:

* Computation of chi-squared detection maps similar to those in [CANDID](https://github.com/amerand/CANDID), but including the possibility to account for correlations. Published in [Kammerer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...644A.110K/abstract) and [Kammerer et al. 2021a](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..36K/abstract).
* MCMC to nail down the companion parameters and estimate their uncertainties. Published in [Kammerer et al. 2021a](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..36K/abstract).
* Karhunen-Loeve calibration (only for kernel-phase). Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).

Future updates will include:

* Karhunen-Loeve calibration (for all types of data). Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Reconstruction of saturated PSFs. Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Modeling of kernel-phase covariance. Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Modeling of GRAVITY covariance. Published in [Kammerer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...644A.110K/abstract).
* Estimation of detection limits.

## Tutorials

There are several tutorials and test data sets provided that can be used to test and explore the functionalities of the code. We note that the code does currently understand OIFITS files and [kernel-phase FITS files](http://frantzmartinache.eu/xara_doc/03_kernel_fits.html).

## HIP 50156 kernel-phase, with covariance, 50 KL components

Chi-squared map and model vs data plot:

![Figure 1](figures/HIP50156_kl50_chi2_map.pdf)
![Figure 2](figures/HIP50156_kl50_kp_bin.pdf)

MCMC chains and posterior:

![Figure 3](figures/HIP50156_kl50_mcmc_chains.pdf)
![Figure 4](figures/HIP50156_kl50_mcmc_corner.pdf)

Chi-squared map and model vs data plot of residuals:

![Figure 5](figures/HIP50156_kl50_sub_chi2_map.pdf)
![Figure 6](figures/HIP50156_kl50_sub_kp_bin.pdf)