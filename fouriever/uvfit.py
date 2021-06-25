from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import emcee
from scipy.optimize import minimize
import sys

from . import inst
from . import plot
from . import util

ftol = 1e-5
rad2mas = 180./np.pi*3600.*1000. # convert rad to mas
mas2rad = np.pi/180./3600./1000. # convert mas to rad
pa_mtoc = '-' # model to chip conversion for position angle


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
            Input directory where the fits files are located.
        fitsfiles: list of str
            List of fits files which shall be opened.
        """
        
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
            List of observables available for the currently selected instrument.
        """
        
        observables_known = ['vis', 'vis2', 't3', 'kp']
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
    
    def gridsearch(self,
                   model='ud_bin',
                   cov=False,
                   sep_range=None,
                   step_size=None,
                   smear=None,
                   ofile=None):
        """
        Parameters
        ----------
        model: str
            Model which shall be fitted.
            Possible values are 'ud', 'bin', 'ud_bin'
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of the grid (mas).
        step_size: float
            Step size of the grid (mas).
        smear: TBD
            
        ofile: str
            Path under which the figure shall be saved.
        
        Returns
        -------
        fit: dict
            Model fit.
        """
        
        data_list = []
        bmax = []
        bmin = []
        dmax = []
        lmin = []
        lerr = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [self.data_list[ww[i]][j]]
                bmax += [np.max(self.data_list[ww[i]][j]['base'])]
                bmin += [np.min(self.data_list[ww[i]][j]['base'])]
                dmax += [self.data_list[ww[i]][j]['diam']]
                lmin += [np.min(self.data_list[ww[i]][j]['wave'])]
                lerr += [np.mean(self.data_list[ww[i]][j]['dwave'])]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['vis2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['vis2v'][:, np.newaxis], wave[np.newaxis, :])
        bmax = np.max(bmax)
        bmin = np.max(bmin)
        dmax = np.max(dmax)
        lmin = np.min(lmin)
        lerr = np.mean(lerr)
        if ('kp' in self.observables):
            smin = 0.5*lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = 5.*lmin/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = lmin/bmin*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Largest spatial scale = %.1f mas' % smax)
        else:
            smin = lmin/bmax*rad2mas # smallest spatial scale (mas)
            waveFOV = lmin**2/lerr/bmax*rad2mas # bandwidth smearing field-of-view (mas)
            diffFOV = 1.2*lmin/dmax*rad2mas # diffraction field-of-view (mas)
            smax = min(waveFOV, diffFOV) # largest spatial scale (mas)
            print('Data properties')
            print('   Smallest spatial scale = %.1f mas' % smin)
            print('   Bandwidth smearing FOV = %.1f mas' % waveFOV)
            print('   Diffraction FOV = %.1f mas' % diffFOV)
            print('   Largest spatial scale = %.1f mas' % smax)
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        if (sep_range is None):
            sep_range = (smin, 1.2*smax)
        if (step_size is None):
            step_size = smin
        
        allcov = True
        ndof = []
        for i in range(len(data_list)):
            if (data_list[i]['covflag'] == False):
                allcov = False
            for j in range(len(self.observables)):
                ndof += [np.prod(data_list[i][self.observables[j]].shape)]
        ndof = np.sum(ndof)
        if (cov == True and allcov == False):
            print('   WARNING: not all data sets have covariances')
        
        if ((model == 'ud') or (model == 'ud_bin')):
            if ('vis2' not in self.observables):
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
            thetae = np.sqrt(thetap['fun']*ftol*np.diag(thetap['hess_inv'].todense()))
            print('   Best fit uniform disk diameter = %.5f +/- %.5f mas' % (thetap['x'][0], thetae))
            print('   Best fit red. chi2 = %.3f (ud)' % (thetap['fun']/ndof))
            fit = {}
            fit['model'] = 'ud'
            fit['p'] = thetap['x']
            fit['dp'] = np.array(thetae)
            fit['chi2_red'] = thetap['fun']/ndof
            fit['ndof'] = ndof
            fit['smear'] = smear
            plot.vis2_ud(data_list=data_list,
                         fit=fit,
                         smear=smear,
                         ofile=ofile)
        else:
            thetap = {}
            thetap['fun'] = util.chi2_ud(p0=np.array([0.]),
                                         data_list=data_list,
                                         observables=self.observables,
                                         cov=cov,
                                         smear=smear)
        
        if ((model == 'bin') or (model == 'ud_bin')):
            if (('t3' not in self.observables) and ('kp' not in self.observables)):
                raise UserWarning('Can only fit companion with closure or kernel phases')
            
            grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range,
                                                     step_size=step_size)
            
            print('Performing grid search (DO NOT TRUST UNCERTAINTIES)')
            f0 = 1e-4
            p0s = []
            pps = []
            pes = []
            chi2s = []
            nc = np.prod(grid_ra_dec[0].shape)
            ctr = 0
            for i in range(grid_ra_dec[0].shape[0]):
                for j in range(grid_ra_dec[0].shape[1]):
                    ctr += 1
                    sys.stdout.write('\r   Cell %.0f of %.0f' % (ctr, nc))
                    sys.stdout.flush()
                    if ((np.isnan(grid_ra_dec[0][i, j]) == False) and (np.isnan(grid_ra_dec[1][i, j]) == False)):
                        if (model == 'bin'):
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
                        pe = np.sqrt(pp['fun']*ftol*np.diag(pp['hess_inv'].todense()))
                        pes += [pe]
                        chi2s += [pp['fun']]
            print('')
            p0s = np.array(p0s)
            pps = np.array(pps)
            pes = np.array(pes)
            chi2s = np.array(chi2s)
            
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
            
            ww = np.argsort(chi2s)
            chi2s_sorted = np.sort(chi2s)
            for i in range(len(chi2s_sorted)):
                pp = pps[ww[i]].copy()
                sep = np.sqrt(pp[1]**2+pp[2]**2)
                if (sep_range[0] <= sep and sep <= sep_range[1]):
                    pa = np.rad2deg(np.arctan2(pp[1], pp[2]))
                    pe = pes[ww[i]].copy()
                    dsep = np.sqrt((pp[1]/sep*pe[1])**2+(pp[2]/sep*pe[2])**2)
                    dpa = np.rad2deg(np.sqrt((pp[2]/sep**2*pe[1])**2+(-pp[1]/sep**2*pe[2])**2))
                    chi2 = chi2s_sorted[i]
                    break
            try:
                nsigma = util.nsigma(chi2r_test=thetap['fun']/ndof,
                                     chi2r_true=chi2/ndof,
                                     ndof=ndof)
            except:
                raise UserWarning('No local minima inside separation range or search box')
            if (model == 'bin'):
                print('   Best fit companion flux = %.3f +/- %.3f %%' % (pp[0]*100., pe[0]*100.))
                print('   Best fit companion right ascension = %.1f +/- %.1f mas' % (pp[1], pe[1]))
                print('   Best fit companion declination = %.1f +/- %.1f mas' % (pp[2], pe[2]))
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
                if ('t3' in self.observables):
                    plot.t3_bin(data_list=data_list,
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
                if ('t3' in self.observables):
                    plot.vis2_t3_ud_bin(data_list=data_list,
                                        fit=fit,
                                        ofile=ofile)
            
            plot.chi2_map(pps_unique=pps_unique,
                          chi2s_unique=chi2s_unique,
                          fit=fit,
                          sep_range=sep_range,
                          step_size=step_size,
                          ofile=ofile)
        
        return fit
    
    def gridsearch_sub(self,
                       fit_sub,
                       model='ud_bin',
                       cov=False,
                       sep_range=None,
                       step_size=None,
                       smear=None,
                       ofile=None):
        """
        Parameters
        ----------
        fit_sub: dict
            Model fit to be subtracted.
        model: str
            Model which shall be fitted.
            Possible values are 'ud', 'bin', 'ud_bin'
        cov: bool
            True if covariance shall be considered.
        sep_range: tuple of float
            Min. and max. angular separation of the grid (mas).
        step_size: float
            Step size of the grid (mas).
        smear: TBD
            
        ofile: str
            Path under which the figure shall be saved.
        
        Returns
        -------
        fit: dict
            Model fit.
        """
        
        print('Subtracting '+fit_sub['model']+' model')
        
        buffer = self.data_list.copy()
        
        if (fit_sub['model'] == 'ud'):
            print('   No companion data found!')
        elif (fit_sub['model'] == 'bin'):
            p0 = fit_sub['p']
        else:
            p0 = fit_sub['p'][:3]
        
        data_before = []
        data_after = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                
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
                p0_temp = np.array([p0[0].copy(), dra_temp, ddec_temp])
                
                vis_sub = util.vis_bin(p0=p0_temp,
                                       data=self.data_list[ww[i]][j],
                                       smear=fit_sub['smear'])
                
                if ('vis' in self.observables):
                    self.data_list[ww[i]][j]['vis'] /= vis_sub
                if ('vis2' in self.observables):
                    self.data_list[ww[i]][j]['vis2'] /= util.vis2vis2(vis_sub)
                if ('t3' in self.observables):
                    data_before += [self.data_list[ww[i]][j]['t3'].copy()]
                    self.data_list[ww[i]][j]['t3'] -= util.vis2t3(vis_sub,
                                                                  data=self.data_list[ww[i]][j])
                    data_after += [self.data_list[ww[i]][j]['t3'].copy()]
                if ('kp' in self.observables):
                    data_before += [self.data_list[ww[i]][j]['kp'].copy()]
                    self.data_list[ww[i]][j]['kp'] -= util.vis2kp(vis_sub,
                                                                  data=self.data_list[ww[i]][j])
                    data_after += [self.data_list[ww[i]][j]['kp'].copy()]
        
        data_before = np.array(data_before)
        data_after = np.array(data_after)
        plt.plot(np.mean(data_before, axis=0), label='before')
        plt.plot(np.mean(data_after, axis=0), label='after')
        plt.xlabel('Index')
        plt.ylabel('Closure/kernel phase')
        plt.title('Binary model subtraction')
        plt.legend()
        plt.show()
        
        fit = self.gridsearch(model=model,
                              cov=cov,
                              sep_range=sep_range,
                              step_size=step_size,
                              smear=smear,
                              ofile=ofile)
        
        self.data_list = buffer
        
        return fit
    
    def mcmc(self,
             fit,
             temp=1.,
             nburn=250,
             nstep=5000,
             cov=False,
             smear=None,
             ofile=None):
        """
        Parameters
        ----------
        fit: dict
            Model fit which shall be explored with the MCMC.
        temp: float
            Covariance inflation factor.
        nburn: int
            Number of burn-in steps for the MCMC to be excluded from the
            posterior distribution.
        nstep: int
            Number of steps for the MCMC to be included in the posterior
            distribution.
        cov: bool
            True if covariance shall be considered.
        smear: TBD
            
        ofile: str
            Path under which the figure shall be saved.
        
        Returns
        -------
        fit: dict
            Model fit.
        """
        
        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.data_list[ww[i]])):
                data_list += [self.data_list[ww[i]][j]]
                if (smear is not None):
                    wave = np.zeros((data_list[-1]['wave'].shape[0]*smear))
                    for k in range(data_list[-1]['wave'].shape[0]):
                        wave[k*smear:(k+1)*smear] = np.linspace(data_list[-1]['wave'][k]-data_list[-1]['dwave'][k], data_list[-1]['wave'][k]+data_list[-1]['dwave'][k], smear)
                    data_list[-1]['uu_smear'] = np.divide(data_list[-1]['vis2u'][:, np.newaxis], wave[np.newaxis, :])
                    data_list[-1]['vv_smear'] = np.divide(data_list[-1]['vis2v'][:, np.newaxis], wave[np.newaxis, :])
        
        allcov = True
        ndof = fit['ndof'].copy()
        for i in range(len(data_list)):
            if (data_list[i]['covflag'] == False):
                allcov = False
        if (cov == True and allcov == False):
            print('   WARNING: not all data sets have covariances')
        
        if (temp is None):
            temp = fit['chi2_red']
        ndim = len(fit['p'])
        nwalkers = (ndim+1)*2
        scale = []
        for i in range(len(fit['p'])):
            if (fit['dp'][i]/fit['p'][i] <= 0.05):
                scale += [fit['dp'][i]]
            else:
                scale += [0.05*fit['p'][i]]
        scale = np.array(scale)
        p0 = [np.random.normal(loc=fit['p'], scale=scale) for i in range(nwalkers)]
        
        if (fit['model'] == 'ud'):
            if ('vis2' not in self.observables):
                raise UserWarning('Can only fit uniform disk with visibility amplitudes')
            print('Computing best fit uniform disk diameter (UNCERTAINTIES FROM MCMC)')
            sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_ud, args=[data_list, self.observables, cov, smear, temp])
        elif (fit['model'] == 'bin'):
            if (('t3' not in self.observables) and ('kp' not in self.observables)):
                raise UserWarning('Can only fit companion with closure or kernel phases')
            print('Computing best fit companion parameters (UNCERTAINTIES FROM MCMC)')
            sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_bin, args=[data_list, self.observables, cov, smear, temp])
        else:
            if ('vis2' not in self.observables):
                raise UserWarning('Can only fit uniform disk with visibility amplitudes')
            if (('t3' not in self.observables) and ('kp' not in self.observables)):
                raise UserWarning('Can only fit companion with closure or kernel phases')
            print('Computing best fit uniform disk and companion parameters (UNCERTAINTIES FROM MCMC)')
            sampler = emcee.EnsembleSampler(nwalkers, ndim, util.lnprob_ud_bin, args=[data_list, self.observables, cov, smear, temp])
        if (smear is not None):
            print('   Bandwidth smearing = %.0f' % smear)
        print('   Covariance inflation factor = %.3f' % temp)
        print('   This may take a few minutes')
        pos, prob, state = sampler.run_mcmc(p0, nburn)
        sampler.reset()
        sampler.run_mcmc(pos, nstep, progress=True)
        
        plot.chains(fit=fit,
                    sampler=sampler,
                    ofile=ofile)
        plot.corner(fit=fit,
                    sampler=sampler,
                    ofile=ofile)
        
        pp = np.percentile(sampler.flatchain, 50., axis=0)
        pu = np.percentile(sampler.flatchain, 84., axis=0)-pp
        pl = pp-np.percentile(sampler.flatchain, 16., axis=0)
        pe = np.mean(np.vstack((pu, pl)), axis=0)
        if (fit['model'] == 'ud'):
            chi2 = util.chi2_ud(p0=pp,
                                data_list=data_list,
                                observables=self.observables,
                                cov=False,
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
        elif (fit['model'] == 'bin'):
            chi2 = util.chi2_bin(p0=pp,
                                 data_list=data_list,
                                 observables=self.observables,
                                 cov=False,
                                 smear=smear)
            chi2_test = util.chi2_ud(p0=np.array([0.]),
                                     data_list=data_list,
                                     observables=self.observables,
                                     cov=cov,
                                     smear=smear)
            nsigma = util.nsigma(chi2r_test=chi2_test/ndof,
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
        else:
            chi2 = util.chi2_ud_bin(p0=pp,
                                    data_list=data_list,
                                    observables=self.observables,
                                    cov=False,
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
            nsigma = util.nsigma(chi2r_test=chi2_test/ndof,
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
        
        return fit
