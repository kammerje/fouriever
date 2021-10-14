# fouriever

A single toolkit for calibrations, correlations, and companion search for kernel-phase, aperture masking, and long-baseline interferometry data.

## Description

This toolkit combines different pieces of code that I developed mainly during my PhD. The major ideas come from [CANDID](https://github.com/amerand/CANDID) and [pynrm](https://github.com/mikeireland/pynrm). Collaborators include **Antoine Merand**, **Mike Ireland**, and **Frantz Martinache**. The ultimate goal is a common framework to analyze and fit kernel-phase, aperture masking, and long-baseline interferometry data. In the near future, a major focus will lie on JWST/NIRCam kernel-phase and JWST/NIRISS aperture masking interferometry.

Currently, the toolkit contains the following functionalities:

* Chi-squared detection maps similar to the fitMaps in [CANDID](https://github.com/amerand/CANDID), but including the possibility to account for correlations. Published in [Kammerer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...644A.110K/abstract) and [Kammerer et al. 2021a](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..36K/abstract).
* Numerically computed bandwidth smearing as in [CANDID](https://github.com/amerand/CANDID).
* [MCMC](https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract) to nail down the companion parameters and estimate their uncertainties. Published in [Kammerer et al. 2021a](https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..36K/abstract).
* Karhunen-Loeve calibration (only for kernel-phase). Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Modeling of aperture masking and long-baseline interferometry correlations. Published in [Kammerer et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...644A.110K/abstract).

Future updates will include:

* Karhunen-Loeve calibration (for all types of data). Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Reconstruction of saturated PSFs. Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Modeling of kernel-phase correlations. Published in [Kammerer et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.486..639K/abstract).
* Estimation of detection limits as in [CANDID](https://github.com/amerand/CANDID).

## Tutorials

There are several tutorials and test data sets provided that can be used to test and explore the functionalities of the fouriever toolkit. We note that the toolkit does currently understand OIFITS files and [kernel-phase FITS files](http://frantzmartinache.eu/xara_doc/03_kernel_fits.html).

## Examples

AX Cir, with correlations and bandwith smearing.

```
Opened PIONIER_Pnat(1.6135391/1.7698610) data
   50 observations
   6 baselines
   4 triangles
   3 wavelengths
Selected instrument = PIONIER_Pnat(1.6135391/1.7698610)
   Use self.set_inst(inst) to change the selected instrument
Selected observables = ['vis2', 't3']
   Use self.set_observables(observables) to change the selected observables
```

Chi-squared map and model vs data plot:

```
Data properties
   Smallest spatial scale = 4.0 mas
   Bandwidth smearing FOV = 69.8 mas
   Diffraction FOV = 221.9 mas
   Largest spatial scale = 69.8 mas
   Bandwidth smearing = 3
   Using data covariance = True
   WARNING: covariance matrix does not have full rank
Computing best fit uniform disk diameter (DO NOT TRUST UNCERTAINTIES)
   Best fit uniform disk diameter = 0.92875 +/- 0.00048 mas
   Best fit red. chi2 = 0.811 (ud)
Computing grid
   Min. sep. = 4.0 mas
   Max. sep. = 83.3 mas
   1352 non-empty grid cells
Computing chi-squared map (DO NOT TRUST UNCERTAINTIES)
   Cell 1849 of 1849
   1319 unique minima found after 1352 fits
   Optimal step size = 3.9 mas
   Current step size = 4.0 mas
   Best fit companion flux = 1.196 +/- 2.823 %
   Best fit companion right ascension = 6.5 +/- 3.5 mas
   Best fit companion declination = -28.2 +/- 0.1 mas
   Best fit companion separation = 29.0 +/- 0.8 mas
   Best fit companion position angle = 167.0 +/- 6.7 deg
   Best fit uniform disk diameter = 0.78203 +/- 0.27875 mas
   Best fit red. chi2 = 0.600 (ud+bin)
   Significance of companion = 9.4 sigma
```

![Figure 1](figures/axcir_smear_cov_chi2map.png)
![Figure 2](figures/axcir_smear_cov_vis2_t3_ud_bin.png)

MCMC chains and posterior:

```
Computing best fit uniform disk and companion parameters (UNCERTAINTIES FROM MCMC)
   Bandwidth smearing = 3
   Using data covariance = True
   WARNING: covariance matrix does not have full rank
   Covariance inflation factor = 0.600
   This may take a few minutes
100%|███████████████████████████████████████| 5000/5000 [05:25<00:00, 15.35it/s]
   Best fit companion flux = 1.096 +/- 0.047 %
   Best fit companion right ascension = 6.2 +/- 0.1 mas
   Best fit companion declination = -28.5 +/- 0.1 mas
   Best fit companion separation = 29.2 +/- 0.1 mas
   Best fit companion position angle = 167.6 +/- 0.1 deg
   Best fit uniform disk diameter = 0.79024 +/- 0.00828 mas
   Best fit red. chi2 = 0.589 (ud+bin)
   Significance of companion = 10.2 sigma
```

![Figure 3](figures/axcir_smear_cov_mcmc_chains.png)
![Figure 4](figures/axcir_smear_cov_mcmc_corner.png)

Chi-squared map and model vs data plot of residuals:

```
Subtracting ud_bin model
Data properties
   Smallest spatial scale = 4.0 mas
   Bandwidth smearing FOV = 69.8 mas
   Diffraction FOV = 221.9 mas
   Largest spatial scale = 69.8 mas
   Bandwidth smearing = 3
   Using data covariance = True
   WARNING: covariance matrix does not have full rank
Computing best fit uniform disk diameter (DO NOT TRUST UNCERTAINTIES)
   Best fit uniform disk diameter = 0.79006 +/- 0.00048 mas
   Best fit red. chi2 = 0.603 (ud)
Computing grid
   Min. sep. = 4.0 mas
   Max. sep. = 83.3 mas
   1352 non-empty grid cells
Computing chi-squared map (DO NOT TRUST UNCERTAINTIES)
   Cell 1849 of 1849
   1349 unique minima found after 1352 fits
   Optimal step size = 4.0 mas
   Current step size = 4.0 mas
   Best fit companion flux = 0.408 +/- 0.744 %
   Best fit companion right ascension = 6.8 +/- 1.7 mas
   Best fit companion declination = 3.0 +/- 1.4 mas
   Best fit companion separation = 7.4 +/- 1.6 mas
   Best fit companion position angle = 66.5 +/- 11.2 deg
   Best fit uniform disk diameter = 0.72003 +/- 0.06686 mas
   Best fit red. chi2 = 0.570 (ud+bin)
   Significance of companion = 1.9 sigma
```

![Figure 5](figures/axcir_smear_cov_sub_chi2map.png)
![Figure 6](figures/axcir_smear_cov_sub_vis2_t3_ud_bin.png)