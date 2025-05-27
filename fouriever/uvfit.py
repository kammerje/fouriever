from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from scipy.linalg import block_diag
from scipy.optimize import minimize

import emcee
import glob
import os
import sys
import warnings

from . import inst
from . import plot
from . import util

rad2mas = 180./np.pi*3600.*1000. # convert rad to mas
mas2rad = np.pi/180./3600./1000. # convert mas to rad
pa_mtoc = '-' # model to chip conversion for position angle
ftol = 1e-5
observables_known = ['v2', 'cp', 'kp']


# =============================================================================
# MAIN
# =============================================================================

class data():
    
    def __init__(self,
                 idir,
                 fitsfiles):
        """
        Parameters
        ----------
        idir: str
            Input directory where fits files are located.
        fitsfiles: list of str, None
            List of fits files which shall be opened. All fits files from
            ``idir`` are opened with ``fitsfiles=None``.
        """
        
        if (fitsfiles is None):
            fitsfiles = glob.glob(idir+'*fits')
            for i, item in enumerate(fitsfiles):
                head, tail = os.path.split(item)
                fitsfiles[i] = tail
        
        self.inst_list = []
        self.data_list = []
        for i in range(len(fitsfiles)):
            inst_list, data_list = inst.open(idir=idir,
                                             fitsfile=fitsfiles[i])
            self.inst_list += inst_list
            self.data_list += data_list
        
        self.set_inst(inst=self.inst_list[0])
        self.set_observables(self.get_observables())
        
        return None
    
    def get_inst(self):
        """
        Returns
        -------
        inst_list: list of str
            List of instruments from which data was opened.
        """
        
        return self.inst_list
    
    def set_inst(self,
                 inst):
        """
        Parameters
        ----------
        inst: str
            Instrument which shall be selected.
        """
        
        if (inst in self.inst_list):
            self.inst = inst
            print('Selected instrument = '+self.inst)
            print('   Use self.set_inst(inst) to change the selected instrument')
        else:
            raise UserWarning(inst+' is an unknown instrument')
        
        return None
    
    def get_observables(self):
        """
        Returns
        -------
        observables: list of str
            List of observables available for currently selected instrument.
        """
        
        observables = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(observables_known)):
            j = 0
            flag = True
            while (j < len(ww) and flag):
                if (observables_known[i] not in self.data_list[ww[j]][0].keys()):
                    flag = False
                j += 1
            if (flag == True):
                observables += [observables_known[i]]
        
        return observables
    
    def set_observables(self,
                        observables):
        """
        Parameters
        ----------
        observables: list of str
            List of observables which shall be selected.
        """
        
        observables_valid = self.get_observables()
        for i in range(len(observables)):
            if (observables[i] not in observables_valid):
                raise UserWarning(observables[i]+' is not a valid observable for the currently selected instrument')
        self.observables = observables
        print('Selected observables = '+str(self.observables))
        print('   Use self.set_observables(observables) to change the selected observables')
        
        return None
    
    def invert(self,
               M):
        """
        Parameters
        ----------
        M: array
            Matrix which shall be inverted.
        
        Returns
        -------
        M_inv: array
            Inverse matrix of M.
        """
        
        sx, sy = M.shape
        if (sx != sy):
            raise UserWarning('Can only invert square matrices')
        M_inv = np.linalg.pinv(M)
        
        return M_inv
    
    def lincmap(self,
                cov=False,
                sep_range=None,
                step_size=None,
                smear=None,
                vmin=None,
                vmax=None,
                ofile=None,
                save_as_fits=False,
                searchbox=None,
                plot_nsigma=False):
        """
        Parameters
        ----------
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of grid (mas).
        step_size: float
            Step size of grid (mas).
        smear: int
            Numerical bandwidth smearing which shall be used.
        vmin : float
            Log10 of contrast map vmin.
        vmax : float
            Log10 of contrast map vmax.
        ofile: str
            Path under which figures shall be saved.
        save_as_fits: bool
            True if result shall be saved as fits file.
        searchbox: dict
            Search box inside of which the companion is expected to be.
            Accepted formats are {'RA': [RA_min, RA_max], 'DEC': [DEC_min,
            DEC_max], 'rho': [rho_min, rho_max], 'phi': [phi_min, phi_max]}.
            Note that -180 <= phi < 180.
        plot_nsigma: bool
            Plot detection significance instead of chi-squared map.
        
        Returns
        -------
        fit: dict
            Best fit model parameters.
        """
        
        if ((len(self.observables) != 1) or ((self.observables[0] != 'cp') and (self.observables[0] != 'kp'))):
            raise UserWarning('Can only compute linear contrast map with closure or kernel phases')
        
        data_list = []
        bmax = []
        bmin = []
        dmax = []
        lmin = []
        lerr = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [deepcopy(self.data_list[ww[i]][j])]
                bmax += [np.max(self.data_list[ww[i]][j]['base'])]
                bmin += [np.min(self.data_list[ww[i]][j]['base'])]
                dmax += [self.data_list[ww[i]][j]['diam']]
                lmin += [np.min(self.data_list[ww[i]][j]['wave'])]
                lerr += [np.mean(self.data_list[ww[i]][j]['dwave'])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        bmax = np.max(bmax)
        bmin = np.max(bmin)
        dmax = np.max(dmax)
        lmin = np.min(lmin)
        lerr = np.mean(lerr)
        if (self.inst in ['NAOS+CONICA', 'NIRC2', 'SPHERE', 'SPHERE-IFS', 'NIRCAM', 'NIRISS', 'ERIS']):
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 5.*lmin/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 0.5*lmin/bmin*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        else:
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 0.5*lmin**2/lerr/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 1.2*lmin/dmax*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Bandwidth smearing FOV = %.1f mas' % waveFOV)
            print('   Diffraction FOV = %.1f mas' % diffFOV)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        if (sep_range is None):
            sep_range = (smin, 1.2*smax)
        if (step_size is None):
            step_size = smin
        
        if (cov == False):
            print('   Using data covariance = False')
            for i in range(len(data_list)):
                data_list[i]['covflag'] = False
        else:
            print('   Using data covariance = True')
            allcov = True
            errflag = False
            for i in range(len(data_list)):
                covs = []
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'v2'):
                        try:
                            covs += [data_list[i]['v2cov']]
                        except:
                            covs += [np.diag(data_list[i]['dv2'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'cp'):
                        try:
                            covs += [data_list[i]['cpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dcp'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'kp'):
                        try:
                            covs += [data_list[i]['kpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dkp'].flatten()**2)]
                            allcov = False
                            covs += []
                data_list[i]['cov'] = block_diag(*covs)
                data_list[i]['icv'] = self.invert(data_list[i]['cov'])
                data_list[i]['covflag'] = True
                if (errflag == False):
                    try:
                        rk = np.linalg.matrix_rank(data_list[i]['cov'])
                        sz = data_list[i]['cov'].shape[0]
                        if (rk < sz):
                            errflag = True
                            print('   WARNING: covariance matrix does not have full rank')
                    except:
                        continue
            if (allcov == False):
                print('   WARNING: not all data sets have covariances')
        
        ndof = []
        for i in range(len(data_list)):
            for j in range(len(self.observables)):
                ndof += [np.prod(data_list[i][self.observables[j]].shape)]
        ndof = np.sum(ndof)
        
        thetap = {}
        thetap['fun'] = util.chi2_bin(p0=np.array([0., 0., 0.]),
                                      data_list=data_list,
                                      observables=self.observables,
                                      cov=cov,
                                      smear=smear)
        
        grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range,
                                                 step_size=step_size,
                                                 verbose=True)
        
        print('Computing linear contrast map (DO NOT TRUST UNCERTAINTIES)')
        f0 = 1e-4
        p0s = []
        pps = []
        pes = []
        chi2s = []
        nsigmas = []
        nc = np.prod(grid_ra_dec[0].shape)
        ctr = 0
        for i in range(grid_ra_dec[0].shape[0]):
            for j in range(grid_ra_dec[0].shape[1]):
                ctr += 1
                if (ctr % 100 == 0):
                    sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
                    sys.stdout.flush()
                if ((np.isnan(grid_ra_dec[0][i, j]) == False) and (np.isnan(grid_ra_dec[1][i, j]) == False)):
                    p0 = np.array([f0, grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]])
                    ff, fe = util.clin(p0,
                                       data_list,
                                       self.observables,
                                       cov,
                                       smear)
                    p0s += [p0]
                    pps += [np.array([ff, grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]])]
                    pes += [np.array([fe, 0., 0.])]
                    chi2s += [util.chi2_bin(pps[-1],
                                            data_list,
                                            self.observables,
                                            cov,
                                            smear)]
                    nsigma, _ = util.nsigma(chi2r_test=thetap['fun']/ndof,
                                            chi2r_true=chi2s[-1]/ndof,
                                            ndof=ndof)
                    nsigmas += [nsigma]
                else:
                    p0s += [np.array([np.nan, np.nan, np.nan])]
                    pps += [np.array([np.nan, np.nan, np.nan])]
                    pes += [np.array([np.nan, np.nan, np.nan])]
                    chi2s += [np.nan]
                    nsigmas += [np.nan]
        sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
        sys.stdout.flush()
        print('')
        p0s = np.array(p0s)
        pps = np.array(pps)
        pes = np.array(pes)
        chi2s = np.array(chi2s)
        nsigmas = np.array(nsigmas)
        
        if (searchbox is None):
            chi2s[pps[:, 0] < 0.] = np.nan
            chi2 = np.nanmin(chi2s)
            pp = pps[np.nanargmin(chi2s)]
            pe = pes[np.nanargmin(chi2s)]
        else:
            chi2s[pps[:, 0] < 0.] = np.nan
            chi2s_copy = chi2s.copy()
            if ('RA' in searchbox.keys()):
                RA = pps[:, 1]
                chi2s_copy[RA < searchbox['RA'][0]] = np.nan
                chi2s_copy[RA > searchbox['RA'][1]] = np.nan
            if ('DEC' in searchbox.keys()):
                DEC = pps[:, 2]
                chi2s_copy[DEC < searchbox['DEC'][0]] = np.nan
                chi2s_copy[DEC > searchbox['DEC'][1]] = np.nan
            if ('rho' in searchbox.keys()):
                rho = np.sqrt(pps[:, 1]**2+pps[:, 2]**2)
                chi2s_copy[rho < searchbox['rho'][0]] = np.nan
                chi2s_copy[rho > searchbox['rho'][1]] = np.nan
            if ('phi' in searchbox.keys()):
                phi = np.rad2deg(np.arctan2(pps[:, 1], pps[:, 2]))
                chi2s_copy[phi < searchbox['phi'][0]] = np.nan
                chi2s_copy[phi > searchbox['phi'][1]] = np.nan
            chi2 = np.nanmin(chi2s_copy)
            pp = pps[np.nanargmin(chi2s_copy)]
            pe = pes[np.nanargmin(chi2s_copy)]
        sep = np.sqrt(pp[1]**2+pp[2]**2)
        pa = np.rad2deg(np.arctan2(pp[1], pp[2]))
        nsigma, _ = util.nsigma(chi2r_test=thetap['fun']/ndof,
                                chi2r_true=chi2/ndof,
                                ndof=ndof)
        
        print('   Best fit companion flux = %.3f +/- %.3f %%' % (pp[0]*100., pe[0]*100.))
        print('   Best fit companion right ascension = %.1f mas' % pp[1])
        print('   Best fit companion declination = %.1f mas' % pp[2])
        print('   Best fit companion separation = %.1f mas' % sep)
        print('   Best fit companion position angle = %.1f deg' % pa)
        print('   Best fit red. chi2 = %.3f (bin)' % (chi2/ndof))
        print('   Significance of companion = %.1f sigma' % nsigma)
        pps = np.swapaxes(np.swapaxes(pps.reshape((grid_ra_dec[0].shape[0], grid_ra_dec[0].shape[1], pps.shape[1])), 0, 2), 1, 2)
        pes = np.swapaxes(np.swapaxes(pes.reshape((grid_ra_dec[0].shape[0], grid_ra_dec[0].shape[1], pes.shape[1])), 0, 2), 1, 2)
        chi2s = chi2s.reshape(grid_ra_dec[0].shape)
        nsigmas = nsigmas.reshape(grid_ra_dec[0].shape)
        
        fit = {}
        fit['model'] = 'bin'
        fit['p'] = pp
        fit['dp'] = pe
        fit['chi2_red'] = chi2/ndof
        fit['ndof'] = ndof
        fit['nsigma'] = nsigma
        fit['smear'] = smear
        fit['cov'] = str(cov)
        fit['pps'] = pps[0]
        fit['pes'] = pes[0]
        fit['chi2s'] = chi2s
        fit['nsigmas'] = nsigmas
        fit['radec'] = grid_ra_dec
        
        plot.lincmap(pps=pps,
                     pes=pes,
                     chi2s=chi2s,
                     nsigmas=nsigmas,
                     fit=fit,
                     sep_range=sep_range,
                     step_size=step_size,
                     vmin=vmin,
                     vmax=vmax,
                     ofile=ofile,
                     searchbox=searchbox,
                     plot_nsigma=plot_nsigma)
        
        if (save_as_fits == True):
            hdu0 = pyfits.PrimaryHDU(pps)
            hdu0.header['EXTNAME'] = 'LINCMAP'
            hdu0.header['MODEL'] = 'bin'
            hdu0.header['NDOF'] = ndof
            hdu0.header['NSIGMA'] = nsigma
            if (smear is None):
                hdu0.header['SMEAR'] = 'None'
            else:
                hdu0.header['SMEAR'] = smear
            hdu0.header['COV'] = str(cov)
            hdu0.header['STEP'] = step_size
            hdu1 = pyfits.ImageHDU(pes)
            hdu1.header['EXTNAME'] = 'DLINCMAP'
            hdu2 = pyfits.ImageHDU(chi2s)
            hdu2.header['EXTNAME'] = 'CHI2MAP'
            hdu3 = pyfits.ImageHDU(nsigmas)
            hdu3.header['EXTNAME'] = 'NSIGMAS'
            hdul = pyfits.HDUList([hdu0, hdu1, hdu2, hdu3])
            hdul.writeto(ofile+'.fits', output_verify='fix', overwrite=True)
            hdul.close()
        
        return fit
    
    def chi2map(self,
                model='ud_bin',
                cov=False,
                sep_range=None,
                step_size=None,
                smear=None,
                ofile=None,
                searchbox=None,
                data_list=None,
                use_mpmath=False):
        """
        Parameters
        ----------
        model: str
            Model which shall be fitted.
            Possible values are 'ud', 'bin', 'ud_bin'.
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of grid (mas).
        step_size: float
            Step size of grid (mas).
        smear: int
            Numerical bandwidth smearing which shall be used.
        ofile: str
            Path under which figures shall be saved.
        searchbox: dict
            Search box inside of which the companion is expected to be.
            Accepted formats are {'RA': [RA_min, RA_max], 'DEC': [DEC_min,
            DEC_max], 'rho': [rho_min, rho_max], 'phi': [phi_min, phi_max]}.
            Note that -180 <= phi < 180.
        data_list: list
            List with the data. Typically the argument can be set to 'None' in
            which case the data is automatically selected from the
            ``data_list`` attribute of the ``data`` class. The parameter is
            internally required by the ``systematics`` method.
        use_mpmath: bool
            Use the ``mpmath`` module for enabling a higher precision
            (50 decimals) on the calculated value from the CDF of the
            chi2 distribution. If set to ``False``, the ``chi2``
            function from ``scipy`` is used. The confidence level is
            always calculated with ``scipy`` and has a maximum value
            of approximately 8 sigma. The default argument is set to
            ``False``.

        Returns
        -------
        fit: dict
            Best fit model parameters.
        """

        # =============================================================
        # Extract data, baseline info and wavelength info from data_list
        # =============================================================
        bmax = []
        bmin = []
        dmax = []
        lmin = []
        lerr = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]  # Use only the datasets's "main" instrument
        if data_list is None:
            data_list = []
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    data_list += [deepcopy(self.data_list[ww[i]][j])]
                    bmax += [np.max(self.data_list[ww[i]][j]['base'])]
                    bmin += [np.min(self.data_list[ww[i]][j]['base'])]
                    dmax += [self.data_list[ww[i]][j]['diam']]
                    lmin += [np.min(self.data_list[ww[i]][j]['wave'])]
                    lerr += [np.mean(self.data_list[ww[i]][j]['dwave'])]
                    if (smear is not None):
                        wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                        for k in range(data_list[-1]['wave'].shape[0]):
                            wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                        data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                        data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        else:
            for j in range(len(data_list)):
                bmax += [np.max(data_list[j]['base'])]
                bmin += [np.min(data_list[j]['base'])]
                dmax += [data_list[j]['diam']]
                lmin += [np.min(data_list[j]['wave'])]
                lerr += [np.mean(data_list[j]['dwave'])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        bmax = np.max(bmax)
        bmin = np.max(bmin)
        dmax = np.max(dmax)
        lmin = np.min(lmin)
        lerr = np.mean(lerr)

        # =============================================================
        # Derive FOV and min/max spatial scale information.
        # Will determine separation range for the grid if not specified
        # =============================================================
        if (self.inst in ['NAOS+CONICA', 'NIRC2', 'SPHERE', 'SPHERE-IFS', 'NIRCAM', 'NIRISS', 'ERIS']):
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 5.*lmin/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 0.5*lmin/bmin*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        else:
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 0.5*lmin**2/lerr/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 1.2*lmin/dmax*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Bandwidth smearing FOV = %.1f mas' % waveFOV)
            print('   Diffraction FOV = %.1f mas' % diffFOV)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        if (sep_range is None):
            sep_range = (smin, 1.2*smax)
        if (step_size is None):
            step_size = smin
        
        # =============================================================
        # Pre-compute data covariance info to data_list
        # =============================================================
        if not cov:
            print('   Using data covariance = False')
            for i in range(len(data_list)):
                data_list[i]['covflag'] = False
        else:
            print('   Using data covariance = True')
            allcov = True
            errflag = False
            for i in range(len(data_list)):
                covs = []
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'v2'):
                        try:
                            covs += [data_list[i]['v2cov']]
                        except Exception:
                            covs += [np.diag(data_list[i]['dv2'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'cp'):
                        try:
                            covs += [data_list[i]['cpcov']]
                        except Exception:
                            covs += [np.diag(data_list[i]['dcp'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'kp'):
                        try:
                            covs += [data_list[i]['kpcov']]
                        except Exception:
                            covs += [np.diag(data_list[i]['dkp'].flatten()**2)]
                            allcov = False
                            covs += []
                data_list[i]['cov'] = block_diag(*covs)
                data_list[i]['icv'] = self.invert(data_list[i]['cov'])
                data_list[i]['covflag'] = True
                if not errflag:
                    try:
                        rk = np.linalg.matrix_rank(data_list[i]['cov'])
                        sz = data_list[i]['cov'].shape[0]
                        if (rk < sz):
                            errflag = True
                            print('   WARNING: covariance matrix does not have full rank')
                    except Exception:
                        continue
            if not allcov:
                print('   WARNING: not all data sets have covariances')
        
        klflag = False
        ndof = []
        for i in range(len(data_list)):
            if data_list[i]['klflag']:
                klflag = True
            for j in range(len(self.observables)):
                ndof += [np.prod(data_list[i][self.observables[j]].shape)]
        ndof = np.sum(ndof)

        # =============================================================
        # Compute best uniform-disk model for chi2 normalization
        # =============================================================
        if ((model == 'ud') or (model == 'ud_bin')):
            # If UD is free in model, optimize
            if ('v2' not in self.observables):
                raise UserWarning('Can only fit uniform disk with visibility amplitudes')
            print('Computing best fit uniform disk diameter (DO NOT TRUST UNCERTAINTIES)')
            theta0 = np.array([1.])
            thetap = minimize(util.chi2_ud,
                              theta0,
                              args=(data_list, self.observables, cov, smear),
                              method='L-BFGS-B',
                              bounds=[(0., np.inf)],
                              tol=ftol,
                              options={'maxiter': 1000})
            thetae = np.sqrt(max(1., abs(thetap['fun']))*ftol*np.diag(thetap['hess_inv'].todense()))
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (thetap['x'][0], thetae))
            print('   Best fit red. chi2 = %.3f (ud)' % (thetap['fun']/ndof))
            fit = {}
            fit['model'] = 'ud'
            fit['p'] = thetap['x']
            fit['dp'] = np.array(thetae)
            fit['chi2_red'] = thetap['fun']/ndof
            fit['ndof'] = ndof
            fit['smear'] = smear
            fit['cov'] = str(cov)
            if klflag:
                plot.v2_ud(data_list=data_list,
                             fit=fit,
                             smear=smear,
                             ofile=ofile)
            else:
                plot.v2_ud_base(data_list=data_list,
                                  fit=fit,
                                  smear=smear,
                                  ofile=ofile)
        else:
            # If UD is not in model, use point-source as reference chi2
            thetap = {}
            thetap['fun'] = util.chi2_ud(p0=np.array([0.]),
                                         data_list=data_list,
                                         observables=self.observables,
                                         cov=cov,
                                         smear=smear)
        
        # =============================================================
        # Chi2 map for binary or binary + uniform disk model
        # =============================================================
        if not ((model == 'bin') or (model == 'ud_bin')):
            raise ValueError(f"Model {model} is not supported for chi2map. Use 'bin' or 'ud_bin'")

        if (('cp' not in self.observables) and ('kp' not in self.observables)):
            raise UserWarning('Can only fit companion with closure or kernel phases')

        grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range,
                                                    step_size=step_size,
                                                    verbose=True)

        print('Computing chi-squared map (DO NOT TRUST UNCERTAINTIES)')
        f0 = 1e-4  # Initial gues for contrast will be 1e-4 at all points
        # List to store minimization starting point
        p0s = []
        # Same as p0s but will also include grid points that are NaN (i.e. outside sep range)
        p0s_all = []
        # List to store minimization result (scipy objects)
        pps = []
        pes = []
        chi2s = []
        nc = np.prod(grid_ra_dec[0].shape)
        ctr = 0
        for i in range(grid_ra_dec[0].shape[0]):
            for j in range(grid_ra_dec[0].shape[1]):
                ctr += 1
                if (ctr % 10 == 0):
                    sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
                    sys.stdout.flush()
                # Ensure current grid point is not nan (i.e. it is within sep range)
                if (not np.isnan(grid_ra_dec[0][i, j])) and (not np.isnan(grid_ra_dec[1][i, j])):
                    if (model == 'bin'):
                        # Initial guess for position is current grid point
                        p0 = np.array([f0, grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]])
                        pp = minimize(util.chi2_bin,
                                        p0,
                                        args=(data_list, self.observables, cov, smear),
                                        method='L-BFGS-B',
                                        bounds=[(0., 1.), (-np.inf, np.inf), (-np.inf, np.inf)],
                                        tol=ftol,
                                        options={'maxiter': 1000})
                    else:
                        p0 = np.array([f0, grid_ra_dec[0][i, j], grid_ra_dec[1][i, j], thetap['x'][0]])
                        pp = minimize(util.chi2_ud_bin,
                                        p0,
                                        args=(data_list, self.observables, cov, smear),
                                        method='L-BFGS-B',
                                        bounds=[(0., 1.), (-np.inf, np.inf), (-np.inf, np.inf), (0., np.inf)],
                                        tol=ftol,
                                        options={'maxiter': 1000})
                    p0s += [p0]
                    pps += [pp['x']]
                    pe = np.sqrt(max(1., abs(pp['fun']))*ftol*np.diag(pp['hess_inv'].todense()))
                    pes += [pe]
                    chi2s += [pp['fun']]
                # Regardless of whether this grid point was NaN or not, still store in p0s_all
                p0 = np.array([f0, grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]])
                p0s_all += [p0]
        sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
        sys.stdout.flush()
        print('')
        p0s = np.array(p0s)
        p0s_all = np.array(p0s_all)
        pps = np.array(pps)
        pes = np.array(pes)
        chi2s = np.array(chi2s)
        
        # For each solution, check its spatial distance from the others.
        # Solutions closer than a certain threshold are considered duplicates
        pps_unique = [pps[0]]
        chi2s_unique = [chi2s[0]]
        dists_unique = []
        for i in range(1, pps.shape[0]):
            diffs = np.array(pps_unique)-pps[i]
            dists = np.sqrt(np.sum(diffs[:, 1:3]**2, axis=1))
            if (model == 'bin'):
                if (np.sum((dists < 0.5*step_size) & (np.abs(diffs[:, 0]) < 1e-2)) > 0):
                    continue
            else:
                if (np.sum((dists < 0.5*step_size) & (np.abs(diffs[:, 0]) < 1e-2) & (np.abs(diffs[:, 3]) < 0.1*smin)) > 0):
                    continue
            pps_unique += [pps[i]]
            chi2s_unique += [chi2s[i]]
            dists_unique += [np.min(dists)]
        pps_unique = np.array(pps_unique)
        chi2s_unique = np.array(chi2s_unique)
        print('   %.0f unique minima found after %.0f fits' % (len(pps_unique), len(pps)))
        opt = np.mean(dists_unique)
        print('   Optimal step size = %.1f mas' % opt)
        print('   Current step size = %.1f mas' % step_size)
        
        # Loop over chi2 values from lowest to highest
        # pick the first one that is inside sep range and search box
        ww = np.argsort(chi2s)
        chi2s_sorted = np.sort(chi2s)
        for i in range(len(chi2s_sorted)):
            pp = pps[ww[i]].copy()
            sep = np.sqrt(pp[1]**2+pp[2]**2)
            if (sep_range[0] <= sep and sep <= sep_range[1]):
                if (searchbox is not None):
                    if ('RA' in searchbox.keys()):
                        RA = pp[1]
                        if ((RA < searchbox['RA'][0]) or (RA > searchbox['RA'][1])):
                            continue
                    if ('DEC' in searchbox.keys()):
                        DEC = pp[2]
                        if ((DEC < searchbox['DEC'][0]) or (DEC > searchbox['DEC'][1])):
                            continue
                    if ('rho' in searchbox.keys()):
                        rho = np.sqrt(pp[1]**2+pp[2]**2)
                        if ((rho < searchbox['rho'][0]) or (rho > searchbox['rho'][1])):
                            continue
                    if ('rho' in searchbox.keys()):
                        phi = np.rad2deg(np.arctan2(pp[1], pp[2]))
                        if ((phi < searchbox['phi'][0]) or (phi > searchbox['phi'][1])):
                            continue
                pa = np.rad2deg(np.arctan2(pp[1], pp[2]))
                pe = pes[ww[i]].copy()
                dsep = np.sqrt((pp[1]/sep*pe[1])**2+(pp[2]/sep*pe[2])**2)
                dpa = np.rad2deg(np.sqrt((pp[2]/sep**2*pe[1])**2+(-pp[1]/sep**2*pe[2])**2))
                chi2 = chi2s_sorted[i]
                break

        # Use the best-fit solution to compute confidence level
        # Based on chi2 ratio with a UD model
        try:
            nsigma, log_bin_prob = util.nsigma(chi2r_test=thetap['fun']/ndof,
                                                chi2r_true=chi2/ndof,
                                                ndof=ndof,
                                                use_mpmath=use_mpmath)
        except Exception:
            raise UserWarning('No local minima inside separation range or search box')

        if (model == 'bin'):
            print('   Best fit companion flux = %.3f +/- %.3f %%' % (pp[0]*100., pe[0]*100.))
            print('   Best fit companion right ascension = %.1f +/- %.1f mas' % (pp[1], pe[1]))
            print('   Best fit companion declination = %.1f +/- %.1f mas' % (pp[2], pe[2]))
            print('   Best fit companion separation = %.1f +/- %.1f mas' % (sep, dsep))
            print('   Best fit companion position angle = %.1f +/- %.1f deg' % (pa, dpa))
            print('   Best fit red. chi2 = %.3f (bin)' % (chi2/ndof))
            print('   Log-probability of companion = %.2e' % log_bin_prob)
            print('   Significance of companion = %.1f sigma' % nsigma)
            fit = {}
            fit['model'] = 'bin'
            fit['p'] = pp
            fit['dp'] = pe
            fit['chi2_red'] = chi2/ndof
            fit['ndof'] = ndof
            fit['log_bin_prob'] = log_bin_prob
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['radec'] = grid_ra_dec
            if ('cp' in self.observables):
                plot.cp_bin(data_list=data_list,
                            fit=fit,
                            smear=smear,
                            ofile=ofile)
            if ('kp' in self.observables):
                plot.kp_bin(data_list=data_list,
                            fit=fit,
                            smear=smear,
                            ofile=ofile)
        else:
            print('   Best fit companion flux = %.3f +/- %.3f %%' % (pp[0]*100., pe[0]*100.))
            print('   Best fit companion right ascension = %.1f +/- %.1f mas' % (pp[1], pe[1]))
            print('   Best fit companion declination = %.1f +/- %.1f mas' % (pp[2], pe[2]))
            print('   Best fit companion separation = %.1f +/- %.1f mas' % (sep, dsep))
            print('   Best fit companion position angle = %.1f +/- %.1f deg' % (pa, dpa))
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (pp[3], pe[3]))
            print('   Best fit red. chi2 = %.3f (ud+bin)' % (chi2/ndof))
            print('   Significance of companion = %.1f sigma' % nsigma)
            fit = {}
            fit['model'] = 'ud_bin'
            fit['p'] = pp
            fit['dp'] = pe
            fit['chi2_red'] = chi2/ndof
            fit['ndof'] = ndof
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
            fit['radec'] = grid_ra_dec
            if ('cp' in self.observables):
                plot.v2_cp_ud_bin(data_list=data_list,
                                    fit=fit,
                                    ofile=ofile)
        
        # Produce chi2 grid by interpolating the minimization results
        # to a fine regular grid
        chi2_map, chi2_grid = plot.chi2map(pps_unique=pps_unique,
                        chi2s_unique=chi2s_unique,
                        fit=fit,
                        sep_range=sep_range,
                        step_size=step_size,
                        ofile=ofile,
                        searchbox=searchbox)

        fit['chi2_map'] = chi2_map
        fit['chi2_grid'] = chi2_grid

        nsigma_map = np.zeros(chi2_map.shape)
        log_prob_map = np.zeros(chi2_map.shape)

        for i in range(chi2_map.shape[0]):
            for j in range(chi2_map.shape[1]):
                nsigma_map[i, j], log_prob_map[i, j] = \
                    util.nsigma(chi2r_test=thetap['fun']/ndof,
                                chi2r_true=chi2_map[i, j]/ndof,
                                ndof=ndof,
                                use_mpmath=use_mpmath)

        fit['nsigma_map'] = nsigma_map
        fit['log_prob_map'] = log_prob_map
        fit['chi2r_test'] = thetap['fun']/ndof

        return fit
    
    def chi2map_sub(self,
                    fit_sub,
                    model='ud_bin',
                    cov=False,
                    sep_range=None,
                    step_size=None,
                    smear=None,
                    ofile=None,
                    searchbox=None):
        """
        Parameters
        ----------
        fit_sub: dict
            Model fit to be subtracted.
        model: str
            Model which shall be fitted.
            Possible values are 'ud', 'bin', 'ud_bin'.
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of grid (mas).
        step_size: float
            Step size of grid (mas).
        smear: int
            Numerical bandwidth smearing which shall be used.
        ofile: str
            Path under which figures shall be saved.
        searchbox: dict
            Search box inside of which the companion is expected to be.
            Accepted formats are {'RA': [RA_min, RA_max], 'DEC': [DEC_min,
            DEC_max], 'rho': [rho_min, rho_max], 'phi': [phi_min, phi_max]}.
            Note that -180 <= phi < 180.
        
        Returns
        -------
        fit: dict
            Best fit model parameters.
        """
        
        print('Subtracting '+fit_sub['model']+' model')
        
        buffer = deepcopy(self.data_list)
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                if (smear is not None):
                    wave = np.zeros((self.data_list[ww[i]][j]['wave'].shape[0]*smear))
                    for k in range(self.data_list[ww[i]][j]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(self.data_list[ww[i]][j]['wave'][k]-0.5*self.data_list[ww[i]][j]['dwave'][k], self.data_list[ww[i]][j]['wave'][k]+0.5*self.data_list[ww[i]][j]['dwave'][k], smear)
                    self.data_list[ww[i]][j]['uu_smear'] = np.divide(self.data_list[ww[i]][j]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    self.data_list[ww[i]][j]['vv_smear'] = np.divide(self.data_list[ww[i]][j]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        
        if (fit_sub['model'] == 'ud'):
            print('   No companion data found!')
        else:
            fit_sub_copy = deepcopy(fit_sub)
            fit_sub_copy['p'][0] = -fit_sub_copy['p'][0]
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                p0 = fit_sub_copy['p']
                dra = p0[1].copy()
                ddec = p0[2].copy()
                rho = np.sqrt(dra**2+ddec**2)
                phi = np.rad2deg(np.arctan2(dra, ddec))
                if (pa_mtoc == '-'):
                    phi -= self.data_list[ww[i]][j]['pa']
                elif (pa_mtoc == '+'):
                    phi += self.data_list[ww[i]][j]['pa']
                else:
                    raise UserWarning('Model to chip conversion for position angle not known')
                phi = ((phi+180.) % 360.)-180.
                dra_temp = rho*np.sin(np.deg2rad(phi))
                ddec_temp = rho*np.cos(np.deg2rad(phi))
                if (fit_sub['model'] == 'bin'):
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                    vis_bin = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                    vis_ref = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                else:
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                    vis_bin = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                    vis_ref = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                
                if ('v2' in self.observables):
                    self.data_list[ww[i]][j]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=self.data_list[ww[i]][j])-util.v2v2(vis_ref, data=self.data_list[ww[i]][j]))
                if ('cp' in self.observables):
                    self.data_list[ww[i]][j]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=self.data_list[ww[i]][j])-util.v2cp(vis_ref, data=self.data_list[ww[i]][j]))
                if ('kp' in self.observables):
                    self.data_list[ww[i]][j]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=self.data_list[ww[i]][j])-util.v2kp(vis_ref, data=self.data_list[ww[i]][j]))
        
        fit = self.chi2map(model=model,
                           cov=cov,
                           sep_range=sep_range,
                           step_size=step_size,
                           smear=smear,
                           ofile=ofile,
                           searchbox=searchbox)
        
        self.data_list = buffer
        
        return fit
    
    def save_sub(self,
                 fit_sub,
                 smear=None,
                 ofile=None):
        """
        Parameters
        ----------
        fit_sub: dict
            Model fit to be subtracted.
        smear: int
            Numerical bandwidth smearing which shall be used.
        ofile: str
            Path under which figures shall be saved.
        
        Returns
        -------
        fit: dict
            Best fit model parameters.
        """
        
        print('Subtracting '+fit_sub['model']+' model')
        
        buffer = deepcopy(self.data_list)
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                if (smear is not None):
                    wave = np.zeros((self.data_list[ww[i]][j]['wave'].shape[0]*smear))
                    for k in range(self.data_list[ww[i]][j]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(self.data_list[ww[i]][j]['wave'][k]-0.5*self.data_list[ww[i]][j]['dwave'][k], self.data_list[ww[i]][j]['wave'][k]+0.5*self.data_list[ww[i]][j]['dwave'][k], smear)
                    self.data_list[ww[i]][j]['uu_smear'] = np.divide(self.data_list[ww[i]][j]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    self.data_list[ww[i]][j]['vv_smear'] = np.divide(self.data_list[ww[i]][j]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        
        if (fit_sub['model'] == 'ud'):
            print('   No companion data found!')
        else:
            fit_sub_copy = deepcopy(fit_sub)
            fit_sub_copy['p'][0] = -fit_sub_copy['p'][0]
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                p0 = fit_sub_copy['p']
                dra = p0[1].copy()
                ddec = p0[2].copy()
                rho = np.sqrt(dra**2+ddec**2)
                phi = np.rad2deg(np.arctan2(dra, ddec))
                if (pa_mtoc == '-'):
                    phi -= self.data_list[ww[i]][j]['pa']
                elif (pa_mtoc == '+'):
                    phi += self.data_list[ww[i]][j]['pa']
                else:
                    raise UserWarning('Model to chip conversion for position angle not known')
                phi = ((phi+180.) % 360.)-180.
                dra_temp = rho*np.sin(np.deg2rad(phi))
                ddec_temp = rho*np.cos(np.deg2rad(phi))
                if (fit_sub['model'] == 'bin'):
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                    vis_bin = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                    vis_ref = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                else:
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                    vis_bin = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                    vis_ref = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                
                if ('v2' in self.observables):
                    self.data_list[ww[i]][j]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=self.data_list[ww[i]][j])-util.v2v2(vis_ref, data=self.data_list[ww[i]][j]))
                if ('cp' in self.observables):
                    self.data_list[ww[i]][j]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=self.data_list[ww[i]][j])-util.v2cp(vis_ref, data=self.data_list[ww[i]][j]))
                if ('kp' in self.observables):
                    self.data_list[ww[i]][j]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=self.data_list[ww[i]][j])-util.v2kp(vis_ref, data=self.data_list[ww[i]][j]))
        
        if ('v2' in self.observables):
            v2_out = []
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    v2_out += [self.data_list[ww[i]][j]['v2']]
            v2_out = np.concatenate(v2_out)
            np.save(ofile+'_v2', v2_out)
        if ('cp' in self.observables):
            cp_out = []
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    cp_out += [self.data_list[ww[i]][j]['cp']]
            cp_out = np.concatenate(cp_out)
            np.save(ofile+'_cp', cp_out)
        if ('kp' in self.observables):
            kp_out = []
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    kp_out += [self.data_list[ww[i]][j]['kp']]
            kp_out = np.concatenate(kp_out)
            np.save(ofile+'_kp', kp_out)
        
        self.data_list = buffer
        
        return None
    
    def mcmc(self,
             fit,
             temp=1.,
             nburn=250,
             nstep=5000,
             nwalkers=None,
             n_live_points=1000,
             sampler='emcee',
             cov=False,
             smear=None,
             ofile=None,
             fixpos=False):
        """
        Parameters
        ----------
        fit: dict
            Best fit model which shall be explored with MCMC.
        temp: float
            Covariance inflation factor.
        nburn: int
            Number of burn-in steps for MCMC to be excluded from posterior
            distribution. This parameter is only used when ``sampler='emcee'``.
        nstep: int
            Number of steps for MCMC to be included in posterior distribution.
        nwalkers: int, None
            Number of walkers. If set to None, the number of walkers
            is set to the default of (ndim+1)*2.
            This parameter is only used when ``sampler='emcee'``.
        n_live_points: int
            Number of live points used for the sampling the posterior
            distribution with ``MultiNest``. This parameter is only used when
            ``sampler='multinest'``.
        sampler: str
            Sampler that is used for the parameter estimation ('emcee' or
            'multinest').
        cov: bool
            True if covariance shall be considered.
        smear: int
            Numerical bandwidth smearing which shall be used.
        ofile: str
            Path under which figures shall be saved.
        fixpos: bool
            Fix position of fit?

        Returns
        -------
        fit: dict
            Best fit model parameters.
        """
        
        if (fit['model'] == 'ud'):
            if ('v2' not in self.observables):
                raise UserWarning('Can only fit uniform disk with visibility amplitudes')
            print('Computing best fit uniform disk diameter (UNCERTAINTIES FROM MCMC)')
        elif (fit['model'] == 'bin'):
            if (('cp' not in self.observables) and ('kp' not in self.observables)):
                raise UserWarning('Can only fit companion with closure or kernel phases')
            print('Computing best fit companion parameters (UNCERTAINTIES FROM MCMC)')
        else:
            if ('v2' not in self.observables):
                raise UserWarning('Can only fit uniform disk with visibility amplitudes')
            if (('cp' not in self.observables) and ('kp' not in self.observables)):
                raise UserWarning('Can only fit companion with closure or kernel phases')
            print('Computing best fit uniform disk and companion parameters (UNCERTAINTIES FROM MCMC)')
        
        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [deepcopy(self.data_list[ww[i]][j])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        
        if (cov == False):
            print('   Using data covariance = False')
            for i in range(len(data_list)):
                data_list[i]['covflag'] = False
        else:
            print('   Using data covariance = True')
            allcov = True
            errflag = False
            for i in range(len(data_list)):
                covs = []
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'v2'):
                        try:
                            covs += [data_list[i]['v2cov']]
                        except:
                            covs += [np.diag(data_list[i]['dv2'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'cp'):
                        try:
                            covs += [data_list[i]['cpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dcp'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'kp'):
                        try:
                            covs += [data_list[i]['kpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dkp'].flatten()**2)]
                            allcov = False
                            covs += []
                data_list[i]['cov'] = block_diag(*covs)
                data_list[i]['icv'] = self.invert(data_list[i]['cov'])
                data_list[i]['covflag'] = True
                if (errflag == False):
                    try:
                        rk = np.linalg.matrix_rank(data_list[i]['cov'])
                        sz = data_list[i]['cov'].shape[0]
                        if (rk < sz):
                            errflag = True
                            print('   WARNING: covariance matrix does not have full rank')
                    except:
                        continue
            if (allcov == False):
                print('   WARNING: not all data sets have covariances')
        
        ndof = []
        for i in range(len(data_list)):
            for j in range(len(self.observables)):
                ndof += [np.prod(data_list[i][self.observables[j]].shape)]
        ndof = np.sum(ndof)
        
        if (fixpos == True):
            pc = fit['p'].copy()
            ec = fit['dp'].copy()
            fit['p'] = np.array([fit['p'][0]])
            fit['dp'] = np.array([fit['dp'][0]])
        
        if (temp is None):
            temp = fit['chi2_red']
        ndim = len(fit['p'])
        if nwalkers is None:
            nwalkers = (ndim+1)*2
        scale = []
        for i in range(len(fit['p'])):
            if (fit['dp'][i]/fit['p'][i] <= 0.05):
                scale += [fit['dp'][i]]
            else:
                scale += [0.05*fit['p'][i]]
        scale = np.array(scale)

        print('   Covariance inflation factor = %.3f' % temp)
        print('   This may take a few minutes')

        if (sampler == 'emcee'):
            p0 = [np.random.normal(loc=fit['p'], scale=scale) for i in range(nwalkers)]

            if (fit['model'] == 'ud'):
                emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_ud, args=[data_list, self.observables, cov, smear, temp])
            elif (fit['model'] == 'bin'):
                if (fixpos == True):
                    emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_bin_fixpos, args=[pc, data_list, self.observables, cov, smear, temp])
                else:
                    emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_bin, args=[data_list, self.observables, cov, smear, temp])
            else:
                emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_ud_bin, args=[data_list, self.observables, cov, smear, temp])

            pos, prob, state = emcee_sampler.run_mcmc(p0, nburn)
            emcee_sampler.reset()
            emcee_sampler.run_mcmc(pos, nstep, progress=True)
            samples = emcee_sampler.get_chain(flat=True)
            ln_z = None

        elif (sampler == 'multinest'):
            # Import here because it will otherwise give a
            # warning if the compiled MultiNest library is
            # not found when importing fouriever
            import pymultinest

            # Get the MPI rank of the process
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ModuleNotFoundError:
                mpi_rank = 0

            # Create the output folder if required
            output_folder = './multinest/'

            if (mpi_rank == 0 and not os.path.exists(output_folder)):
                os.mkdir(output_folder)

            # Set uniform prior boundaries with respect
            # to best-fit value from the chi2map
            prior_bounds = []
            for i, item in enumerate(fit['p']):
                if i == (len(fit['p']) - 2):
                    # RA range (mas)
                    prior_bounds.append((item-20., item+20.))
                elif i == (len(fit['p']) - 1):
                    # Dec range (mas)
                    prior_bounds.append((item-20., item+20.))
                else:
                    # Contrast range
                    prior_bounds.append((0., 1.))

            def lnprior_multinest(cube, n_dim, n_param):
                """
                Parameters
                ----------
                cube : np.ndarray
                    Unit cube.
                n_dim : int
                    Number of dimensions. This parameter is mandatory.
                n_param : int
                    Number of parameters. This parameter is mandatory.

                Returns
                -------
                np.ndarray
                    Array with model parameters.
                """

                for i in range(n_param):
                    cube[i] = (prior_bounds[i][0] + (prior_bounds[i][1] - prior_bounds[i][0]) * cube[i])

                return cube

            def lnprob_multinest(cube, n_dim, n_param) -> np.float64:
                """
                Parameters
                ----------
                cube : pymultinest.run.LP_c_double
                    Cube with physical parameters.
                n_dim : int
                    Number of dimensions. This parameter is mandatory.
                n_param : int
                    Number of parameters. This parameter is mandatory.

                Returns
                -------
                float
                    Log-likelihood.
                """

                params = np.zeros(n_param)
                for i in range(n_param):
                    params[i] = cube[i]

                if (fit['model'] == 'ud'):
                    return util.lnprob_ud(params, data_list, self.observables, cov=cov, smear=smear, temp=temp)

                elif (fit['model'] == 'bin'):
                    return util.lnprob_bin(params, data_list, self.observables, cov=cov, smear=smear, temp=temp)

                else:
                    return util.lnprob_ud_bin(params, data_list, self.observables, cov=cov, smear=smear, temp=temp)

            pymultinest.run(
                lnprob_multinest,
                lnprior_multinest,
                len(fit['p']),
                outputfiles_basename=output_folder,
                resume=False,
                n_live_points=n_live_points,
            )

            analyzer = pymultinest.analyse.Analyzer(len(fit['p']), outputfiles_basename=output_folder)
            sampling_stats = analyzer.get_stats()
            samples = analyzer.get_equal_weighted_posterior()

            # Nested sampling global log-evidence
            ln_z = sampling_stats["nested importance sampling global log-evidence"]
            ln_z_error = sampling_stats["nested importance sampling global log-evidence error"]
            print(f"   Log-evidence: {ln_z:.2f} +/- {ln_z_error:.2f}")

            ln_prob = samples[:, -1]
            samples = samples[:, :-1]

        if (sampler == 'emcee'):
            plot.chains(fit=fit,
                        samples=samples,
                        ofile=ofile,
                        fixpos=fixpos)

        if ofile is not None:
            plot.corner(fit=fit,
                        samples=samples,
                        ofile=ofile,
                        fixpos=fixpos)
        
        pp = np.percentile(samples, 50., axis=0)
        pu = np.percentile(samples, 84., axis=0)-pp
        pl = pp-np.percentile(samples, 16., axis=0)
        pe = np.mean(np.vstack((pu, pl)), axis=0)
        if (fit['model'] == 'ud'):
            chi2 = util.chi2_ud(p0=pp,
                                data_list=data_list,
                                observables=self.observables,
                                cov=cov,
                                smear=smear)
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (pp[0], pe[0]))
            print('   Best fit red. chi2 = %.3f (ud)' % (chi2/ndof))
            fit = {}
            fit['model'] = 'ud'
            fit['p'] = pp
            fit['dp'] = pe
            fit['chi2_red'] = chi2/ndof
            fit['ndof'] = ndof
            fit['smear'] = smear
            fit['cov'] = str(cov)
        elif (fit['model'] == 'bin'):
            if (fixpos == True):
                pp = np.append(pp, pc[1:])
                pe = np.append(pe, ec[1:])
            chi2 = util.chi2_bin(p0=pp,
                                 data_list=data_list,
                                 observables=self.observables,
                                 cov=cov,
                                 smear=smear)
            chi2_test = util.chi2_ud(p0=np.array([0.]),
                                     data_list=data_list,
                                     observables=self.observables,
                                     cov=cov,
                                     smear=smear)
            nsigma, _ = util.nsigma(chi2r_test=chi2_test/ndof,
                                    chi2r_true=chi2/ndof,
                                    ndof=ndof)
            sep = np.sqrt(pp[-2]**2+pp[-1]**2)
            pa = np.rad2deg(np.arctan2(pp[-2], pp[-1]))
            dsep = np.sqrt((pp[-2]/sep*pe[-2])**2+(pp[-1]/sep*pe[-1])**2)
            dpa = np.rad2deg(np.sqrt((pp[-1]/sep**2*pe[-2])**2+(-pp[-2]/sep**2*pe[-1])**2))
            print('   Best fit companion flux = %.3f +/- %.3f %%' % (np.mean(pp[:-2])*100., np.mean(pe[:-2])*100.))
            print('   Best fit companion right ascension = %.1f +/- %.1f mas' % (pp[-2], pe[-2]))
            print('   Best fit companion declination = %.1f +/- %.1f mas' % (pp[-1], pe[-1]))
            print('   Best fit companion separation = %.1f +/- %.1f mas' % (sep, dsep))
            print('   Best fit companion position angle = %.1f +/- %.1f deg' % (pa, dpa))
            print('   Best fit red. chi2 = %.3f (bin)' % (chi2/ndof))
            print('   Significance of companion = %.1f sigma' % nsigma)
            fit = {}
            fit['model'] = 'bin'
            fit['p'] = pp
            fit['dp'] = pe
            fit['chi2_red'] = chi2/ndof
            fit['ndof'] = ndof
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)
        else:
            chi2 = util.chi2_ud_bin(p0=pp,
                                    data_list=data_list,
                                    observables=self.observables,
                                    cov=cov,
                                    smear=smear)
            theta0 = np.array([1.])
            thetap = minimize(util.chi2_ud,
                              theta0,
                              args=(data_list, self.observables, cov, smear),
                              method='L-BFGS-B',
                              bounds=[(0., np.inf)],
                              tol=ftol,
                              options={'maxiter': 1000})
            chi2_test = util.chi2_ud(p0=thetap['x'],
                                     data_list=data_list,
                                     observables=self.observables,
                                     cov=cov,
                                     smear=smear)
            nsigma, _ = util.nsigma(chi2r_test=chi2_test/ndof,
                                    chi2r_true=chi2/ndof,
                                    ndof=ndof)
            sep = np.sqrt(pp[1]**2+pp[2]**2)
            pa = np.rad2deg(np.arctan2(pp[1], pp[2]))
            dsep = np.sqrt((pp[1]/sep*pe[1])**2+(pp[2]/sep*pe[2])**2)
            dpa = np.rad2deg(np.sqrt((pp[2]/sep**2*pe[1])**2+(-pp[1]/sep**2*pe[2])**2))
            print('   Best fit companion flux = %.3f +/- %.3f %%' % (pp[0]*100., pe[0]*100.))
            print('   Best fit companion right ascension = %.1f +/- %.1f mas' % (pp[1], pe[1]))
            print('   Best fit companion declination = %.1f +/- %.1f mas' % (pp[2], pe[2]))
            print('   Best fit companion separation = %.1f +/- %.1f mas' % (sep, dsep))
            print('   Best fit companion position angle = %.1f +/- %.1f deg' % (pa, dpa))
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (pp[3], pe[3]))
            print('   Best fit red. chi2 = %.3f (ud+bin)' % (chi2/ndof))
            print('   Significance of companion = %.1f sigma' % nsigma)
            fit = {}
            fit['model'] = 'ud_bin'
            fit['p'] = pp
            fit['dp'] = pe
            fit['chi2_red'] = chi2/ndof
            fit['ndof'] = ndof
            fit['nsigma'] = nsigma
            fit['smear'] = smear
            fit['cov'] = str(cov)

        fit['sampler'] = sampler
        fit['samples'] = samples
        if ln_z is not None:
            fit['ln-evidence'] = (ln_z, ln_z_error)

        return fit
    
    def detlim(self,
               sigma=3.,
               fit_sub=None,
               fit_sub_2=None,
               cov=False,
               sep_range=None,
               step_size=None,
               smear=None,
               ofile=None,
               cmin=1e-6):
        """
        Parameters
        ----------
        sigma: int
            Confidence level for which the detection limits shall be computed.
        fit_sub: dict
            Model fit to be subtracted.
        fit_sub_2: dict
            Second model fit to be subtracted.
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of grid (mas).
        step_size: float
            Step size of grid (mas).
        smear: int
            Numerical bandwidth smearing which shall be used.
        ofile: str
            Path under which figures shall be saved.
        cmin: float
            Minimum contrast for which you want to fit.
        """
        
        if (fit_sub is not None):
            print('Subtracting '+fit_sub['model']+' model')
            
            buffer = deepcopy(self.data_list)
            
            ww = np.where(np.array(self.inst_list) == self.inst)[0]
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    if (smear is not None):
                        wave = np.zeros((self.data_list[ww[i]][j]['wave'].shape[0]*smear))
                        for k in range(self.data_list[ww[i]][j]['wave'].shape[0]):
                            wave[k*smear:(k+1)*smear] = np.linspace(self.data_list[ww[i]][j]['wave'][k]-0.5*self.data_list[ww[i]][j]['dwave'][k], self.data_list[ww[i]][j]['wave'][k]+0.5*self.data_list[ww[i]][j]['dwave'][k], smear)
                        self.data_list[ww[i]][j]['uu_smear'] = np.divide(self.data_list[ww[i]][j]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                        self.data_list[ww[i]][j]['vv_smear'] = np.divide(self.data_list[ww[i]][j]['v2v'][:, np.newaxis], wave[np.newaxis, :])
            
            if (fit_sub['model'] == 'ud'):
                print('   No companion data found!')
            else:
                fit_sub_copy = deepcopy(fit_sub)
                fit_sub_copy['p'][0] = -fit_sub_copy['p'][0]
            
            ww = np.where(np.array(self.inst_list) == self.inst)[0]
            for i in range(len(ww)):
                for j in range(len(self.data_list[ww[i]])):
                    p0 = fit_sub_copy['p']
                    dra = p0[1].copy()
                    ddec = p0[2].copy()
                    rho = np.sqrt(dra**2+ddec**2)
                    phi = np.rad2deg(np.arctan2(dra, ddec))
                    if (pa_mtoc == '-'):
                        phi -= self.data_list[ww[i]][j]['pa']
                    elif (pa_mtoc == '+'):
                        phi += self.data_list[ww[i]][j]['pa']
                    else:
                        raise UserWarning('Model to chip conversion for position angle not known')
                    phi = ((phi+180.) % 360.)-180.
                    dra_temp = rho*np.sin(np.deg2rad(phi))
                    ddec_temp = rho*np.cos(np.deg2rad(phi))
                    if (fit_sub['model'] == 'bin'):
                        p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                        vis_bin = util.vis_bin(p0=p0_temp,
                                               data=self.data_list[ww[i]][j],
                                               smear=fit_sub['smear'])
                        p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                        vis_ref = util.vis_bin(p0=p0_temp,
                                               data=self.data_list[ww[i]][j],
                                               smear=fit_sub['smear'])
                    else:
                        p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                        vis_bin = util.vis_ud_bin(p0=p0_temp,
                                                  data=self.data_list[ww[i]][j],
                                                  smear=fit_sub['smear'])
                        p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                        vis_ref = util.vis_ud_bin(p0=p0_temp,
                                                  data=self.data_list[ww[i]][j],
                                                  smear=fit_sub['smear'])
                    
                    if ('v2' in self.observables):
                        self.data_list[ww[i]][j]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=self.data_list[ww[i]][j])-util.v2v2(vis_ref, data=self.data_list[ww[i]][j]))
                    if ('cp' in self.observables):
                        self.data_list[ww[i]][j]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=self.data_list[ww[i]][j])-util.v2cp(vis_ref, data=self.data_list[ww[i]][j]))
                    if ('kp' in self.observables):
                        self.data_list[ww[i]][j]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=self.data_list[ww[i]][j])-util.v2kp(vis_ref, data=self.data_list[ww[i]][j]))
        
            if (fit_sub_2 is not None):
                print('Subtracting '+fit_sub_2['model']+' model')
                
                ww = np.where(np.array(self.inst_list) == self.inst)[0]
                for i in range(len(ww)):
                    for j in range(len(self.data_list[ww[i]])):
                        if (smear is not None):
                            wave = np.zeros((self.data_list[ww[i]][j]['wave'].shape[0]*smear))
                            for k in range(self.data_list[ww[i]][j]['wave'].shape[0]):
                                wave[k*smear:(k+1)*smear] = np.linspace(self.data_list[ww[i]][j]['wave'][k]-0.5*self.data_list[ww[i]][j]['dwave'][k], self.data_list[ww[i]][j]['wave'][k]+0.5*self.data_list[ww[i]][j]['dwave'][k], smear)
                            self.data_list[ww[i]][j]['uu_smear'] = np.divide(self.data_list[ww[i]][j]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                            self.data_list[ww[i]][j]['vv_smear'] = np.divide(self.data_list[ww[i]][j]['v2v'][:, np.newaxis], wave[np.newaxis, :])
                
                if (fit_sub_2['model'] == 'ud'):
                    print('   No companion data found!')
                else:
                    fit_sub_2_copy = deepcopy(fit_sub_2)
                    fit_sub_2_copy['p'][0] = -fit_sub_2_copy['p'][0]
                
                ww = np.where(np.array(self.inst_list) == self.inst)[0]
                for i in range(len(ww)):
                    for j in range(len(self.data_list[ww[i]])):
                        p0 = fit_sub_2_copy['p']
                        dra = p0[1].copy()
                        ddec = p0[2].copy()
                        rho = np.sqrt(dra**2+ddec**2)
                        phi = np.rad2deg(np.arctan2(dra, ddec))
                        if (pa_mtoc == '-'):
                            phi -= self.data_list[ww[i]][j]['pa']
                        elif (pa_mtoc == '+'):
                            phi += self.data_list[ww[i]][j]['pa']
                        else:
                            raise UserWarning('Model to chip conversion for position angle not known')
                        phi = ((phi+180.) % 360.)-180.
                        dra_temp = rho*np.sin(np.deg2rad(phi))
                        ddec_temp = rho*np.cos(np.deg2rad(phi))
                        if (fit_sub_2['model'] == 'bin'):
                            p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                            vis_bin = util.vis_bin(p0=p0_temp,
                                                   data=self.data_list[ww[i]][j],
                                                   smear=fit_sub_2['smear'])
                            p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                            vis_ref = util.vis_bin(p0=p0_temp,
                                                   data=self.data_list[ww[i]][j],
                                                   smear=fit_sub_2['smear'])
                        else:
                            p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                            vis_bin = util.vis_ud_bin(p0=p0_temp,
                                                      data=self.data_list[ww[i]][j],
                                                      smear=fit_sub_2['smear'])
                            p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                            vis_ref = util.vis_ud_bin(p0=p0_temp,
                                                      data=self.data_list[ww[i]][j],
                                                      smear=fit_sub_2['smear'])
                        
                        if ('v2' in self.observables):
                            self.data_list[ww[i]][j]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=self.data_list[ww[i]][j])-util.v2v2(vis_ref, data=self.data_list[ww[i]][j]))
                        if ('cp' in self.observables):
                            self.data_list[ww[i]][j]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=self.data_list[ww[i]][j])-util.v2cp(vis_ref, data=self.data_list[ww[i]][j]))
                        if ('kp' in self.observables):
                            self.data_list[ww[i]][j]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=self.data_list[ww[i]][j])-util.v2kp(vis_ref, data=self.data_list[ww[i]][j]))
        
        data_list = []
        bmax = []
        bmin = []
        dmax = []
        lmin = []
        lerr = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [deepcopy(self.data_list[ww[i]][j])]
                bmax += [np.max(self.data_list[ww[i]][j]['base'])]
                bmin += [np.min(self.data_list[ww[i]][j]['base'])]
                dmax += [self.data_list[ww[i]][j]['diam']]
                lmin += [np.min(self.data_list[ww[i]][j]['wave'])]
                lerr += [np.mean(self.data_list[ww[i]][j]['dwave'])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        bmax = np.max(bmax)
        bmin = np.max(bmin)
        dmax = np.max(dmax)
        lmin = np.min(lmin)
        lerr = np.mean(lerr)
        if (self.inst in ['NAOS+CONICA', 'NIRC2', 'SPHERE', 'SPHERE-IFS', 'NIRCAM', 'NIRISS', 'ERIS']):
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 5.*lmin/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 0.5*lmin/bmin*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        else:
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 0.5*lmin**2/lerr/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 1.2*lmin/dmax*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Bandwidth smearing FOV = %.1f mas' % waveFOV)
            print('   Diffraction FOV = %.1f mas' % diffFOV)
            print('   Largest spatial scale = %.1f mas' % smax)
            print('   Minimum baseline = %.2f m' % bmin)
            print('   Maximum baseline = %.2f m' % bmax)
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        if (sep_range is None):
            sep_range = (smin, 1.2*smax)
        if (step_size is None):
            step_size = smin
        
        if (cov == False):
            print('   Using data covariance = False')
            for i in range(len(data_list)):
                data_list[i]['covflag'] = False
        else:
            print('   Using data covariance = True')
            allcov = True
            errflag = False
            for i in range(len(data_list)):
                covs = []
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'v2'):
                        try:
                            covs += [data_list[i]['v2cov']]
                        except:
                            covs += [np.diag(data_list[i]['dv2'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'cp'):
                        try:
                            covs += [data_list[i]['cpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dcp'].flatten()**2)]
                            allcov = False
                            covs += []
                    if (self.observables[j] == 'kp'):
                        try:
                            covs += [data_list[i]['kpcov']]
                        except:
                            covs += [np.diag(data_list[i]['dkp'].flatten()**2)]
                            allcov = False
                            covs += []
                data_list[i]['cov'] = block_diag(*covs)
                data_list[i]['icv'] = self.invert(data_list[i]['cov'])
                data_list[i]['covflag'] = True
                if (errflag == False):
                    try:
                        rk = np.linalg.matrix_rank(data_list[i]['cov'])
                        sz = data_list[i]['cov'].shape[0]
                        if (rk < sz):
                            errflag = True
                            print('   WARNING: covariance matrix does not have full rank')
                    except:
                        continue
            if (allcov == False):
                print('   WARNING: not all data sets have covariances')
        
        klflag = False
        ndof = []
        for i in range(len(data_list)):
            if (data_list[i]['klflag'] == True):
                klflag = True
            for j in range(len(self.observables)):
                ndof += [np.prod(data_list[i][self.observables[j]].shape)]
        ndof = np.sum(ndof)
        
        if ('v2' in self.observables):
            print('Computing best fit uniform disk diameter (DO NOT TRUST UNCERTAINTIES)')
            theta0 = np.array([1.])
            thetap = minimize(util.chi2_ud,
                              theta0,
                              args=(data_list, self.observables, cov, smear),
                              method='L-BFGS-B',
                              bounds=[(0., np.inf)],
                              tol=ftol,
                              options={'maxiter': 1000})
            thetae = np.sqrt(max(1., abs(thetap['fun']))*ftol*np.diag(thetap['hess_inv'].todense()))
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (thetap['x'][0], thetae))
            print('   Best fit red. chi2 = %.3f (ud)' % (thetap['fun']/ndof))
        else:
            thetap = {}
            thetap['fun'] = util.chi2_ud(p0=np.array([0.]),
                                         data_list=data_list,
                                         observables=self.observables,
                                         cov=cov,
                                         smear=smear)
        
        if (('cp' not in self.observables) and ('kp' not in self.observables)):
            raise UserWarning('Can only compute detection limits with closure or kernel phases')
        
        grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range,
                                                 step_size=step_size,
                                                 verbose=True)
        
        sigma = int(sigma)
        print('Computing detection limits ('+str(sigma)+'-sigma)')
        if (cmin >= 1.):
            raise ValueError('Minimum contrast must be less than 1')
        f0s = np.logspace(np.log10(cmin), 0, 200)
        ffs_absil = []
        ffs_injection = []
        nc = np.prod(grid_ra_dec[0].shape)
        ctr = 0
        for i in range(grid_ra_dec[0].shape[0]):
            for j in range(grid_ra_dec[0].shape[1]):
                ctr += 1
                if (ctr % 10 == 0):
                    sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
                    sys.stdout.flush()
                if ((np.isnan(grid_ra_dec[0][i, j]) == False) and (np.isnan(grid_ra_dec[1][i, j]) == False)):
                    
                    # Absil method.
                    if ('v2' in self.observables):
                        p0 = np.array([f0s[0], grid_ra_dec[0][i, j], grid_ra_dec[1][i, j], thetap['x'][0]])
                        temp = [self.lim_absil(f0, util.chi2_ud_bin, p0, data_list, self.observables, cov, smear, thetap['fun'], ndof, sigma) for f0 in f0s]
                        temp = np.array(temp)
                        f0 = f0s[np.argmin(temp)]
                        pp = minimize(self.lim_absil,
                                      f0,
                                      args=(util.chi2_ud_bin, p0, data_list, self.observables, cov, smear, thetap['fun'], ndof, sigma),
                                      method='L-BFGS-B',
                                      bounds=[(0., 1.)],
                                      tol=ftol,
                                      options={'maxiter': 1000})
                    else:
                        p0 = np.array([f0s[0], grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]])
                        temp = [self.lim_absil(f0, util.chi2_bin, p0, data_list, self.observables, cov, smear, thetap['fun'], ndof, sigma) for f0 in f0s]
                        temp = np.array(temp)
                        f0 = f0s[np.argmin(temp)]
                        pp = minimize(self.lim_absil,
                                      f0,
                                      args=(util.chi2_bin, p0, data_list, self.observables, cov, smear, thetap['fun'], ndof, sigma),
                                      method='L-BFGS-B',
                                      bounds=[(0., 1.)],
                                      tol=ftol,
                                      options={'maxiter': 1000})
                    ffs_absil += [pp['x'][0].copy()]
                    
                    # Injection method.
                    if ('v2' in self.observables):
                        fit_inj = {'p': np.array([f0s[0], grid_ra_dec[0][i, j], grid_ra_dec[1][i, j], 0.]),
                                    'model': 'bin',
                                    'smear': smear}
                        temp = [self.lim_injection(f0, fit_inj, data_list, self.observables, cov, smear, ndof, thetap['x'][0], sigma) for f0 in f0s]
                        temp = np.array(temp)
                        f0 = f0s[np.argmin(temp)]
                        pp = minimize(self.lim_injection,
                                      f0,
                                      args=(fit_inj, data_list, self.observables, cov, smear, ndof, thetap['x'][0], sigma),
                                      method='L-BFGS-B',
                                      bounds=[(0., 1.)],
                                      tol=ftol,
                                      options={'maxiter': 1000})
                    else:
                        # fit_inj = {'p': np.array([f0s[0], grid_ra_dec[0][i, j], grid_ra_dec[1][i, j]]),
                        #             'model': 'bin',
                        #             'smear': smear}
                        # temp = [self.lim_injection(f0, fit_inj, data_list, self.observables, cov, smear, ndof, None, sigma) for f0 in f0s]
                        # temp = np.array(temp)
                        # f0 = f0s[np.argmin(temp)]
                        # pp = minimize(self.lim_injection,
                        #               f0,
                        #               args=(fit_inj, data_list, self.observables, cov, smear, ndof, None, sigma),
                        #               method='L-BFGS-B',
                        #               bounds=[(0., 1.)],
                        #               tol=ftol,
                        #               options={'maxiter': 1000})
                        pp = {'x': [ffs_absil[-1]]}
                    ffs_injection += [pp['x'][0].copy()]
                
                else:
                    ffs_absil += [np.nan]
                    ffs_injection += [np.nan]
        sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
        sys.stdout.flush()
        print('')
        
        ffs_absil = np.array(ffs_absil).reshape(grid_ra_dec[0].shape)
        ffs_injection = np.array(ffs_injection).reshape(grid_ra_dec[0].shape)
        
        plot.detlim(ffs_absil, ffs_injection, sigma, sep_range, step_size, ofile)
        
        if (fit_sub is not None):
            self.data_list = buffer
        
        pass
    
    def lim_absil(self,
                  f0,
                  func,
                  p0,
                  data_list,
                  observables,
                  cov,
                  smear,
                  chi2_true,
                  ndof,
                  sigma=3):
        """
        Parameters
        ----------
        f0: float
            Relative flux of companion.
        func: method
            Method to compute chi-squared.
        p0: array
            p0[0]: float
                Relative flux of companion.
            p0[1]: float
                Right ascension offset of companion.
            p0[2]: float
                Declination offset of companion.
            p0[3]: float
                Uniform disk diameter (mas).
        data_list: list of dict
            List of data whose chi-squared shall be computed. The list
            contains one data structure for each observation.
        observables: list of str
            List of observables which shall be considered.
        cov: bool
            True if covariance shall be considered.
        smear: int
            Numerical bandwidth smearing which shall be used.
        chi2r_true: float
            Reduced chi-squared of true model.
        ndof: int
            Number of degrees of freedom.
        sigma: int
            Confidence level for which the detection limits shall be computed.
        
        Returns
        -------
        chi2: float
            Chi-squared of Absil method.
        """
        
        if (f0 <= 0.):
            
            return np.inf
        else:
            pp = p0.copy()
            pp[0] = f0
            chi2_test = func(p0=pp,
                             data_list=data_list,
                             observables=observables,
                             cov=cov,
                             smear=smear)
            nsigma, _ = util.nsigma(chi2r_test=chi2_test/ndof,
                                    chi2r_true=chi2_true/ndof,
                                    ndof=ndof)
            
            return np.abs(nsigma-sigma)**2
    
    def lim_injection(self,
                      f0,
                      fit_inj,
                      data_list,
                      observables,
                      cov,
                      smear,
                      ndof,
                      thetap=None,
                      sigma=3):
        """
        Parameters
        ----------
        f0: float
            Relative flux of companion.
        fit_inj: dict
            Model fit to be injected.
        data_list: list of dict
            List of data whose chi-squared shall be computed. The list
            contains one data structure for each observation.
        observables: list of str
            List of observables which shall be considered.
        cov: bool
            True if covariance shall be considered.
        smear: int
            Numerical bandwidth smearing which shall be used.
        ndof: int
            Number of degrees of freedom.
        thetap: float
            
        sigma: int
            Confidence level for which the detection limits shall be computed.
        
        Returns
        -------
        chi2: float
            Chi-squared of Injection method.
        """
        
        if (f0 <= 0.):
            
            return np.inf
        else:
            fit_inj_copy = deepcopy(fit_inj)
            fit_inj_copy['p'][0] = f0
            data_list_copy = deepcopy(data_list)
            data_list_copy = self.inj_companion(data_list=data_list_copy,
                                                fit_inj=fit_inj_copy)
            
            if ('v2' in observables):
                thetap_ud = minimize(util.chi2_ud,
                                     np.array([thetap]),
                                     args=(data_list_copy, observables, cov, smear),
                                     method='L-BFGS-B',
                                     bounds=[(0., np.inf)],
                                     tol=ftol,
                                     options={'maxiter': 1000})
                chi2_ud = thetap_ud['fun']
                fit_inj_copy['p'][3] = thetap
                chi2_bin = util.chi2_ud_bin(p0=fit_inj_copy['p'],
                                            data_list=data_list_copy,
                                            observables=observables,
                                            cov=cov,
                                            smear=smear)
            else:
                chi2_ud = util.chi2_ud(p0=np.array([0.]),
                                       data_list=data_list_copy,
                                       observables=observables,
                                       cov=cov,
                                       smear=smear)
                chi2_bin = util.chi2_bin(p0=fit_inj_copy['p'],
                                         data_list=data_list_copy,
                                         observables=observables,
                                         cov=cov,
                                         smear=smear)
            nsigma, _ = util.nsigma(chi2r_test=chi2_ud/ndof,
                                    chi2r_true=chi2_bin/ndof,
                                    ndof=ndof)
            
            return np.abs(nsigma-sigma)**2
    
    def inj_companion(self,
                      data_list,
                      fit_inj):
        """
        Parameters
        ----------
        data_list: list
            Subset of the self.data_list data structure into which the
            companion shall be injected.
        fit_inj: dict
            Model fit to be injected.
        """
        
        if (fit_inj['model'] == 'ud'):
            print('   No companion data found!')
        
        for i in range(len(data_list)):
            p0 = fit_inj['p']
            dra = p0[1].copy()
            ddec = p0[2].copy()
            rho = np.sqrt(dra**2+ddec**2)
            phi = np.rad2deg(np.arctan2(dra, ddec))
            if (pa_mtoc == '-'):
                phi -= data_list[i]['pa']
            elif (pa_mtoc == '+'):
                phi += data_list[i]['pa']
            else:
                raise UserWarning('Model to chip conversion for position angle not known')
            phi = ((phi+180.) % 360.)-180.
            dra_temp = rho*np.sin(np.deg2rad(phi))
            ddec_temp = rho*np.cos(np.deg2rad(phi))
            if (fit_inj['model'] == 'bin'):
                p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                vis_bin = util.vis_bin(p0=p0_temp,
                                       data=data_list[i],
                                       smear=fit_inj['smear'])
                p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                vis_ref = util.vis_bin(p0=p0_temp,
                                       data=data_list[i],
                                       smear=fit_inj['smear'])
            else:
                p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                vis_bin = util.vis_ud_bin(p0=p0_temp,
                                          data=data_list[i],
                                          smear=fit_inj['smear'])
                p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                vis_ref = util.vis_ud_bin(p0=p0_temp,
                                          data=data_list[i],
                                          smear=fit_inj['smear'])
            
            if ('v2' in self.observables):
                data_list[i]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=data_list[i])-util.v2v2(vis_ref, data=data_list[i]))
            if ('cp' in self.observables):
                data_list[i]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=data_list[i])-util.v2cp(vis_ref, data=data_list[i]))
            if ('kp' in self.observables):
                data_list[i]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=data_list[i])-util.v2kp(vis_ref, data=data_list[i]))
        
        return data_list
    
    def sub_companion(self,
                      fit_sub):
        """
        Parameters
        ----------
        fit_sub: dict
            Model fit to be subtracted.
        """
        
        print('Subtracting '+fit_sub['model']+' model')
        
        buffer = deepcopy(self.data_list)
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                if (fit_sub['smear'] is not None):
                    wave = np.zeros((self.data_list[ww[i]][j]['wave'].shape[0]*fit_sub['smear']))
                    for k in range(self.data_list[ww[i]][j]['wave'].shape[0]):
                        wave[k*fit_sub['smear']:(k+1)*fit_sub['smear']] = np.linspace(self.data_list[ww[i]][j]['wave'][k]-0.5*self.data_list[ww[i]][j]['dwave'][k], self.data_list[ww[i]][j]['wave'][k]+0.5*self.data_list[ww[i]][j]['dwave'][k], fit_sub['smear'])
                    self.data_list[ww[i]][j]['uu_smear'] = np.divide(self.data_list[ww[i]][j]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    self.data_list[ww[i]][j]['vv_smear'] = np.divide(self.data_list[ww[i]][j]['v2v'][:, np.newaxis], wave[np.newaxis, :])
        
        if (fit_sub['model'] == 'ud'):
            print('   No companion data found!')
        else:
            fit_sub_copy = deepcopy(fit_sub)
            fit_sub_copy['p'][0] = -fit_sub_copy['p'][0]
        
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                p0 = fit_sub_copy['p']
                dra = p0[1].copy()
                ddec = p0[2].copy()
                rho = np.sqrt(dra**2+ddec**2)
                phi = np.rad2deg(np.arctan2(dra, ddec))
                if (pa_mtoc == '-'):
                    phi -= self.data_list[ww[i]][j]['pa']
                elif (pa_mtoc == '+'):
                    phi += self.data_list[ww[i]][j]['pa']
                else:
                    raise UserWarning('Model to chip conversion for position angle not known')
                phi = ((phi+180.) % 360.)-180.
                dra_temp = rho*np.sin(np.deg2rad(phi))
                ddec_temp = rho*np.cos(np.deg2rad(phi))
                if (fit_sub['model'] == 'bin'):
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp]) # w/ companion
                    vis_bin = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp]) # w/o companion
                    vis_ref = util.vis_bin(p0=p0_temp,
                                           data=self.data_list[ww[i]][j],
                                           smear=fit_sub['smear'])
                else:
                    p0_temp = np.array([np.abs(p0[0].copy()), dra_temp, ddec_temp, p0[3].copy()]) # w/ companion
                    vis_bin = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                    p0_temp = np.array([0., dra_temp, ddec_temp, p0[3].copy()]) # w/o companion
                    vis_ref = util.vis_ud_bin(p0=p0_temp,
                                              data=self.data_list[ww[i]][j],
                                              smear=fit_sub['smear'])
                
                if ('v2' in self.observables):
                    self.data_list[ww[i]][j]['v2'] += np.sign(p0[0])*(util.v2v2(vis_bin, data=self.data_list[ww[i]][j])-util.v2v2(vis_ref, data=self.data_list[ww[i]][j]))
                if ('cp' in self.observables):
                    self.data_list[ww[i]][j]['cp'] += np.sign(p0[0])*(util.v2cp(vis_bin, data=self.data_list[ww[i]][j])-util.v2cp(vis_ref, data=self.data_list[ww[i]][j]))
                if ('kp' in self.observables):
                    self.data_list[ww[i]][j]['kp'] += np.sign(p0[0])*(util.v2kp(vis_bin, data=self.data_list[ww[i]][j])-util.v2kp(vis_ref, data=self.data_list[ww[i]][j]))
        
        return buffer
    
    def systematics(self,
                    fit,
                    pa_step=1.,
                    n_remove=None,
                    smear=None,
                    ofile=None):
        """
        Method for estimating the systematic uncertainties by retrieving the
        contrast and position of artificial companions that are step-wise
        injected at a range of position angles. Only the binary model is
        supported.
        
        Parameters
        ----------
        fit: dict
            Best fit model of which the parameters will be used to subtract
            a real companion from the data. The returned dictionary from
            ``chi2map`` should be used as argument for the ``fit`` parameter.
        pa_step: float
            Step size (in deg) of the position angle, going from 0 to 360 deg,
            at which injection-retrieval test will be done. The default
            argument is set to 1 deg so that a total of 360 samples will be
            created.
        n_remove: int, None
            Number of real companions to remove from the data before
            estimating the systematic uncertainties. These are assumed to be
            the brightest point sources in the data. No companions are removed
            if the argument is set to 'None'.
        smear: int, None
            Numerical bandwidth smearing which shall be used. The recommended
            value is 3. By default the argument is set to 'None' so smearing
            is not corrected for.
        ofile: str, None
            Path under which figures shall be saved. No figures are stored if
            the argument is set to 'None'.
        
        Returns
        -------
        offset: np.ndarray
            Array that contains the offsets between the injected and retrieved
            parameters at the tested position angles. The shape of the array
            is (n_samples, 3) with the 3 columns being the flux ratio, delta
            RA (mas), and delta Dec (mas).
        """
        
        # Check if a binary model was used.
        fit_model = fit['model']
        if fit_model != 'bin':
            raise ValueError(f'The model of the fit dictionary is set to {fit_model} while only a binary model (model=bin) is supported.')
        
        # Check if the same smear value is used as with chi2map.
        fit_smear = fit['smear']
        if smear != fit_smear:
            warnings.warn(f'The argument of the smear parameter is set to {smear} while the value of the smear keyword in the fit dictionary is set to {fit_smear}.')
        
        # Select the data.
        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [deepcopy(self.data_list[ww[i]][j])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])

        # Adopt separation range and step size from chi2map result.
        sep_range = (np.nanmin(np.abs(fit['radec'][0][np.nonzero(fit['radec'][0])])),
                     np.nanmax(np.abs(fit['radec'][0])))
        step_size = np.abs(np.nanmean(np.diff(fit['radec'][0])))
        
        # Remove real companion(s) from the data.
        fit_comp = deepcopy(fit)
        data_comp = deepcopy(data_list)
        if n_remove is not None:
            for i in range(n_remove):
                
                # Inject the negative companion model.
                fit_copy = deepcopy(fit_comp)
                fit_copy['p'][0] = -fit_copy['p'][0]
                data_comp = self.inj_companion(data_list=deepcopy(data_comp),
                                               fit_inj=fit_copy)
                
                # Create plot to check if the companion is removed.
                if ofile is None:
                    out_file = None
                else:
                    out_file = f'{ofile}_removed_{i}'
                fit_comp = self.chi2map(model='bin',
                                        cov=fit['cov'],
                                        sep_range=sep_range,
                                        step_size=step_size,
                                        smear=smear,
                                        ofile=out_file,
                                        searchbox=None,
                                        data_list=data_comp)
        
        # Separation and PAs for injection-recovery test.
        sep_test = np.sqrt(fit['p'][1]**2+fit['p'][2]**2) # mas
        pa_test = np.arange(0., 360., pa_step) # deg
        
        # Create empty array for storing the result.
        offset = np.zeros((pa_test.size, 3))
        
        # Iterate over PAs in steps of step_size deg.
        for pa_idx, pa_item in enumerate(pa_test):
            
            # Convert sep-PA to RA-Dec for artificial source. Use the
            # separation and contrast of the actual companion that has been
            # removed from the data.
            fit_test = deepcopy(fit)
            fit_test['p'][1] = sep_test*np.sin(np.radians(pa_item))
            fit_test['p'][2] = sep_test*np.cos(np.radians(pa_item))
            
            # Inject artificial source.
            data_test = self.inj_companion(data_list=deepcopy(data_comp),
                                           fit_inj=fit_test)
            
            # Retrieve the contrast and position.
            searchbox = {'RA': (fit_test['p'][1]-10., fit_test['p'][1]+10.),
                         'DEC': (fit_test['p'][2]-10., fit_test['p'][2]+10.)}
            if ofile is None:
                out_file = None
            else:
                out_file = f'{ofile}_injected_{pa_idx}'
            fit_chi2 = self.chi2map(model='bin',
                                    cov=fit['cov'],
                                    sep_range=(sep_test-20., sep_test+20.),
                                    step_size=step_size,
                                    smear=smear,
                                    ofile=out_file,
                                    searchbox=searchbox,
                                    data_list=data_test)
            
            # Store the offset between the injected and retrieved values of
            # the binary parameters.
            offset[pa_idx, 0] = fit_test['p'][0] - fit_chi2['p'][0]
            offset[pa_idx, 1] = fit_test['p'][1] - fit_chi2['p'][1]
            offset[pa_idx, 2] = fit_test['p'][2] - fit_chi2['p'][2]
        
        return offset

    def estimate_phase(self,
                       fit=None,
                       smear=None,
                       ofile=None,
                       scatter_kwargs=None):
        """
        Method for estimating the phases from the closure phases.
        
        Parameters
        ----------
        fit: dict, None
            Best-fit parameters for the binary model. The returned
            dictionary from ``chi2map`` can be used as argument for the
            ``fit`` parameter. The binary model is not plotted if the
            argument is set to 'None'.
        smear: int, None
            Numerical bandwidth smearing which shall be used. The recommended
            value is 3. By default the argument is set to 'None' so smearing
            is not corrected for.
        ofile: str, None
            Path under which the figure will be stored. No plot is created if
            the argument is set to 'None'.
        scatter_kwargs: dict, None
            Optional dictionary with keyword arguments for the `scatter plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html>`_
            of the phases that are extracted from the data. Default values
            are used for the plot if the argument is set to 'None'.

        Returns
        -------
        phase_list: list(np.ndarray)
            List with arrays that contain the phases in degrees. The length
            of the list is equal to the number of data files. The size of
            each array is equal to the number of baselines.
        u_list: list(np.ndarray)
            List with arrays that contain the u coordinates in 1/arcsec.
            Multiply with 180/pi*3600 to convert to B/lambda.
        u_list: list(np.ndarray)
            List with arrays that contain the v coordinates in 1/arcsec.
            Multiply with 180/pi*3600 to convert to B/lambda.
        """

        # Radians to arcsec conversion factor
        rad2asec = 180./np.pi*3600.

        # Set kwargs dictionary with default values for scatter plot
        if scatter_kwargs is None:
            scatter_kwargs = {'marker': 's', 'lw': 0.5, 'alpha': 0.5}

        # Select the data
        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [deepcopy(self.data_list[ww[i]][j])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-0.5*data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+0.5*data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['v2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['v2v'][:, np.newaxis], wave[np.newaxis, :])

        # Initiate the output list that will be returned
        u_list = []
        v_list = []
        phase_list = []

        # Initiate model lists for plotting
        model_u = None
        model_v = None
        model_phase = None
        u_comp = None
        v_comp = None

        # Iterate over the data files
        for data_idx, data_item in enumerate(data_list):
            if fit is not None and ofile is not None and data_idx == 0:
                # Calculate phase with the binary model and best-fit parameters
                # uu and vv are defined as baseline (m) divided by wavelength (m)
                uv_max = max(np.amax(np.abs(data_item['uu'])), np.amax(np.abs(data_item['vv'])))
                u = np.linspace(1.2*uv_max, -1.2*uv_max, 101)
                v = np.linspace(-1.2*uv_max, 1.2*uv_max, 101)
                uu, vv = np.meshgrid(u, v)

                # Create a data dictionary with the u-v grid
                data = {'uu': uu, 'vv': vv}

                # Calculate the model visibilities for the best-fit-parameters
                vis = util.vis_bin(fit['p'], data)

                # Extract the phase from the complex visibilities
                model_phase = np.degrees(np.angle(vis))

                # Convert from B/lambda to 1/arcsec
                u /= rad2asec
                v /= rad2asec

                # Set the (arbitrary) extent of the arrow that indicates
                # the direction to the companion of the provided parameters
                u_comp = 0.5*(uv_max/rad2asec)*fit['p'][1]/np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
                v_comp = 0.5*(uv_max/rad2asec)*fit['p'][2]/np.sqrt(fit['p'][1]**2+fit['p'][2]**2)

            # Invert the CP-to-phase matrix
            cpmat = data_item['cpmat']
            cpmat_inv = np.linalg.pinv(cpmat)

            # Project the closure phases and convert to degrees
            phase = cpmat_inv @ data_item['cp'][:, 0]
            phase = np.degrees(phase)

            # Extract the u and v coordinates
            u_coord = data_item['uu'][:, 0]
            v_coord = data_item['vv'][:, 0]

            # Convert from B/lambda to 1/arcsec
            u_coord /= rad2asec
            v_coord /= rad2asec


            if ofile is not None:
                # Create scatter plot of phases in the u-v plane. Positive
                # phase are plotted in orange and negative phases in gray.
                plt.scatter(u_coord[phase<0.], v_coord[phase<0.], c='none',
                            s=40.*np.abs(phase[phase<0.]), edgecolor='silver', **scatter_kwargs)
                plt.scatter(u_coord[phase>0.], v_coord[phase>0.], c='none',
                            s=40.*phase[phase>0.], edgecolor='tab:orange', **scatter_kwargs)

                # Due to the anti-symmetry of the phases, the colors are
                # swapped on the mirrored side
                plt.scatter(-u_coord[phase<0.], -v_coord[phase<0.], c='none',
                            s=40.*np.abs(phase[phase<0.]), edgecolor='tab:orange', **scatter_kwargs)
                plt.scatter(-u_coord[phase>0.], -v_coord[phase>0.], c='none',
                            s=40.*phase[phase>0.], edgecolor='silver', **scatter_kwargs)

            # Add the phases and coordinates to the output lists
            phase_list.append(phase)
            u_list.append(u_coord)
            v_list.append(v_coord)

        plot.estimate_phase(
            phase_list,
            u_list,
            v_list,
            model_phase=model_phase,
            model_u=model_u,
            model_v=model_v,
            scatter_kwargs=scatter_kwargs,
            u_comp=u_comp,
            v_comp=v_comp,
            ofile=ofile,
        )

        return phase_list, u_list, v_list
