from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import mpmath
import numpy as np

from scipy import stats
from scipy.interpolate import interp1d
from scipy.special import j1


rad2mas = 180./np.pi*3600.*1000. # convert rad to mas
mas2rad = np.pi/180./3600./1000. # convert mas to rad
pa_mtoc = '-' # model to chip conversion for position angle


# =============================================================================
# MAIN
# =============================================================================

def get_grid(sep_range,
             step_size,
             verbose=True):
    """
    Parameters
    ----------
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    verbose: bool
        True if feedback shall be printed.
    
    Returns
    -------
    grid_ra_dec: tuple of array
        grid_ra_dec[0]: array
            Right ascension offset of grid cells (mas).
        grid_ra_dec[1]: array
            Declination offset of grid cells (mas).
    grid_sep_pa: tuple of array
        grid_sep_pa[0]: array
            Angular separation of grid cells (mas).
        grid_sep_pa[1]: array
            Position angle of grid cells (deg).
    """
    
    if (verbose == True):
        print('Computing grid')
    
    nc = int(np.ceil(sep_range[1]/step_size))
    temp = np.linspace(-nc*step_size, nc*step_size, 2*nc+1)
    grid_ra_dec = np.meshgrid(temp, temp)
    grid_ra_dec[0] = np.fliplr(grid_ra_dec[0])
    sep = np.sqrt(grid_ra_dec[0]**2+grid_ra_dec[1]**2)
    pa = np.rad2deg(np.arctan2(grid_ra_dec[0], grid_ra_dec[1]))
    grid_sep_pa = np.array([sep, pa])
    
    mask = (sep < sep_range[0]-1e-6) | (sep_range[1]+1e-6 < sep)
    grid_ra_dec[0][mask] = np.nan
    grid_ra_dec[1][mask] = np.nan
    grid_sep_pa[0][mask] = np.nan
    grid_sep_pa[1][mask] = np.nan
    
    if (verbose == True):
        print('   Min. sep. = %.1f mas' % np.nanmin(grid_sep_pa[0]))
        print('   Max. sep. = %.1f mas' % np.nanmax(grid_sep_pa[0]))
        print('   %.0f non-empty grid cells' % np.sum(np.logical_not(np.isnan(grid_sep_pa[0]))))
    
    return grid_ra_dec, grid_sep_pa

def v2v2(vis,
         data):
    """
    Parameters
    ----------
    vis: array
        Complex visibility.
    data: dict
        Data structure.
    
    Returns
    -------
    v2: array
        Squared visibility amplitude.
    """
    
    if (data['klflag'] == True):
        return np.abs(np.dot(data['v2mat'], vis))**2
    else:
        return np.abs(vis)**2

def v2cp(vis,
         data):
    """
    Parameters
    ----------
    vis: array
        Complex visibility.
    data: dict
        Data structure.
    
    Returns
    -------
    cp: array
        Closure phase (rad).
    """
    
    return np.dot(data['cpmat'], np.angle(vis))

def v2kp(vis,
         data):
    """
    Parameters
    ----------
    vis: array
        Complex visibility.
    data: dict
        Data structure.
    
    Returns
    -------
    kp: array
        Kernel phase (rad).
    """
    
    return np.dot(data['kpmat'], np.angle(vis))

def clin(p0,
         data_list,
         observables,
         cov=False,
         smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    ff: float
        Best fit relative companion flux.
    fe: float
        Error of best fit relative companion flux.
    """
    
    mod_icv_sig = []
    mod_icv_mod = []
    for i in range(len(data_list)):
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
        p0_temp = np.array([p0[0].copy(), dra_temp, ddec_temp])
        
        vis_mod = vis_bin(p0=p0_temp,
                          data=data_list[i],
                          smear=smear)
        sig = []
        err = []
        mod = []
        for j in range(len(observables)):
            if (observables[j] == 'cp'):
                sig += [data_list[i]['cp']]
                err += [data_list[i]['dcp']]
                mod += [v2cp(vis_mod,
                               data=data_list[i])/p0[0]]
            elif (observables[j] == 'kp'):
                sig += [data_list[i]['kp']]
                err += [data_list[i]['dkp']]
                mod += [v2kp(vis_mod,
                               data=data_list[i])/p0[0]]
        sig = np.concatenate(sig).flatten()
        mod = np.concatenate(mod).flatten()
        if (cov == False):
            var = np.concatenate(err).flatten()**2
            mod_icv = np.divide(mod, var)
        else:
            if (data_list[i]['covflag'] == False):
                var = np.concatenate(err).flatten()**2
                mod_icv = np.divide(mod, var)
            else:
                mod_icv = mod.dot(data_list[i]['icv'])
        mod_icv_sig += [mod_icv.dot(sig)]
        mod_icv_mod += [mod_icv.dot(mod)]
    ff = np.sum(mod_icv_sig)/np.sum(mod_icv_mod)
    fe = 1./np.sum(mod_icv_mod)
    
    return ff, fe

def vis_ud(p0,
           data,
           smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Uniform disk diameter (mas).
    data: dict
        Data structure.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    vis: array
        Complex visibility of uniform disk.
    """
    
    if (smear is None):
        vis = np.pi*p0[0]*mas2rad*np.sqrt(data['uu']**2+data['vv']**2)
        vis += 1e-6*(vis == 0)
        vis = 2.*j1(vis)/vis
    else:
        vis = np.pi*p0[0]*mas2rad*np.sqrt(data['uu_smear']**2+data['vv_smear']**2)
        vis += 1e-6*(vis == 0)
        vis = 2.*j1(vis)/vis
        
        vis = vis.reshape((vis.shape[0], vis.shape[1]//smear, smear))
        vis = np.mean(vis, axis=2)
    
    return vis

def chi2_ud(p0,
            data_list,
            observables,
            cov=False,
            smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Uniform disk diameter (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    chi2: array
        Chi-squared of uniform disk model.
    """
    
    chi2 = []
    for i in range(len(data_list)):
        vis_mod = vis_ud(p0=p0,
                         data=data_list[i],
                         smear=smear)
        sig = []
        err = []
        mod = []
        for j in range(len(observables)):
            if (observables[j] == 'v2'):
                sig += [data_list[i]['v2']]
                err += [data_list[i]['dv2']]
                mod += [v2v2(vis_mod,
                                 data=data_list[i])]
            elif (observables[j] == 'cp'):
                sig += [data_list[i]['cp']]
                err += [data_list[i]['dcp']]
                mod += [v2cp(vis_mod,
                               data=data_list[i])]
            elif (observables[j] == 'kp'):
                sig += [data_list[i]['kp']]
                err += [data_list[i]['dkp']]
                mod += [v2kp(vis_mod,
                               data=data_list[i])]
        sig = np.concatenate(sig).flatten()
        mod = np.concatenate(mod).flatten()
        res = sig-mod
        if (cov == False):
            var = np.concatenate(err).flatten()**2
            res_icv = np.divide(res, var)
        else:
            if (data_list[i]['covflag'] == False):
                var = np.concatenate(err).flatten()**2
                res_icv = np.divide(res, var)
            else:
                res_icv = res.dot(data_list[i]['icv'])
        chi2 += [res_icv.dot(res)]
    
    return np.sum(chi2)

def lnprob_ud(p0,
              data_list,
              observables,
              cov=False,
              smear=None,
              temp=1.):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Uniform disk diameter (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    temp: float
        Covariance inflation factor.
    
    Returns
    -------
    lnprob: float
        Log-likelihood of uniform disk model.
    """
    
    if (p0[0] < 0.):
        
        return -np.inf
    
    else:
        chi2 = chi2_ud(p0,
                       data_list,
                       observables,
                       cov=cov,
                       smear=smear)
        
        return -0.5*np.sum(chi2)/temp

def vis_bin(p0,
            data,
            smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
    data: dict
        Data structure.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    vis: array
        Complex visibility of unresolved companion.
    """
    
    if (smear is None):
        v1 = 1.0+0.0j
        v2 = 1.0+0.0j
        vis = v2*p0[0]*np.exp(-2.*np.pi*1.0j*mas2rad*(data['uu']*p0[1]+data['vv']*p0[2]))
        vis = (v1+vis)/(1.+p0[0])
    else:
        v1 = 1.0+0.0j
        v2 = 1.0+0.0j
        vis = v2*p0[0]*np.exp(-2.*np.pi*1.0j*mas2rad*(data['uu_smear']*p0[1]+data['vv_smear']*p0[2]))
        vis = (v1+vis)/(1.+p0[0])
        
        vis = vis.reshape((vis.shape[0], vis.shape[1]//smear, smear))
        vis = np.mean(vis, axis=2)
    
    return vis

def chi2_bin(p0,
             data_list,
             observables,
             cov=False,
             smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    chi2: array
        Chi-squared of unresolved companion model.
    """
    
    if (len(p0) > 3):
        wavel_index = {}
        wavel_count = 0
        
        chi2 = []
        for i in range(len(data_list)):
            dra = p0[-2].copy()
            ddec = p0[-1].copy()
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
            if data_list[i]['wave'][0] not in wavel_index:
                wavel_index[data_list[i]['wave'][0]] = wavel_count
                wavel_count += 1
            p0_temp = np.array([p0[wavel_index[data_list[i]['wave'][0]]].copy(), dra_temp, ddec_temp])
            
            vis_mod = vis_bin(p0=p0_temp,
                              data=data_list[i],
                              smear=smear)
            sig = []
            err = []
            mod = []
            for j in range(len(observables)):
                if (observables[j] == 'v2'):
                    sig += [data_list[i]['v2']]
                    err += [data_list[i]['dv2']]
                    mod += [v2v2(vis_mod,
                                     data=data_list[i])]
                elif (observables[j] == 'cp'):
                    sig += [data_list[i]['cp']]
                    err += [data_list[i]['dcp']]
                    mod += [v2cp(vis_mod,
                                   data=data_list[i])]
                elif (observables[j] == 'kp'):
                    sig += [data_list[i]['kp']]
                    err += [data_list[i]['dkp']]
                    mod += [v2kp(vis_mod,
                                   data=data_list[i])]
            sig = np.concatenate(sig).flatten()
            mod = np.concatenate(mod).flatten()
            res = sig-mod
            if (cov == False):
                var = np.concatenate(err).flatten()**2
                res_icv = np.divide(res, var)
            else:
                if (data_list[i]['covflag'] == False):
                    var = np.concatenate(err).flatten()**2
                    res_icv = np.divide(res, var)
                else:
                    res_icv = res.dot(data_list[i]['icv'])
            chi2 += [res_icv.dot(res)]
        
        return np.sum(chi2)
    
    else:
        
        chi2 = []
        for i in range(len(data_list)):
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
            p0_temp = np.array([p0[0].copy(), dra_temp, ddec_temp])
            
            vis_mod = vis_bin(p0=p0_temp,
                              data=data_list[i],
                              smear=smear)
            sig = []
            err = []
            mod = []
            for j in range(len(observables)):
                if (observables[j] == 'v2'):
                    sig += [data_list[i]['v2']]
                    err += [data_list[i]['dv2']]
                    mod += [v2v2(vis_mod,
                                     data=data_list[i])]
                elif (observables[j] == 'cp'):
                    sig += [data_list[i]['cp']]
                    err += [data_list[i]['dcp']]
                    mod += [v2cp(vis_mod,
                                   data=data_list[i])]
                elif (observables[j] == 'kp'):
                    sig += [data_list[i]['kp']]
                    err += [data_list[i]['dkp']]
                    mod += [v2kp(vis_mod,
                                   data=data_list[i])]
            sig = np.concatenate(sig).flatten()
            mod = np.concatenate(mod).flatten()
            res = sig-mod
            if (cov == False):
                var = np.concatenate(err).flatten()**2
                res_icv = np.divide(res, var)
            else:
                if (data_list[i]['covflag'] == False):
                    var = np.concatenate(err).flatten()**2
                    res_icv = np.divide(res, var)
                else:
                    res_icv = res.dot(data_list[i]['icv'])
            chi2 += [res_icv.dot(res)]
        
        return np.sum(chi2)

def lnprob_bin(p0,
               data_list,
               observables,
               cov=False,
               smear=None,
               temp=1.):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    temp: float
        Covariance inflation factor.
    
    Returns
    -------
    lnprob: float
        Log-likelihood of unresolved companion model.
    """
    
    if ((p0[0] < 0.) or (np.abs(p0[1]) > 10000.) or (np.abs(p0[2]) > 10000.)):
        
        return -np.inf
    
    chi2 = chi2_bin(p0,
                    data_list,
                    observables,
                    cov=cov,
                    smear=smear)
    
    return -0.5*np.sum(chi2)/temp

def vis_ud_bin(p0,
               data,
               smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
        p0[3]: float
            Uniform disk diameter (mas).
    data: dict
        Data structure.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    vis: array
        Complex visibility of uniform disk with unresolved companion.
    """
    
    if (smear is None):
        v1 = np.pi*p0[3]*mas2rad*np.sqrt(data['uu']**2+data['vv']**2)
        v1 += 1e-6*(v1 == 0)
        v1 = 2.*j1(v1)/v1
        v2 = 1.0+0.0j
        vis = v2*p0[0]*np.exp(-2.*np.pi*1.0j*mas2rad*(data['uu']*p0[1]+data['vv']*p0[2]))
        vis = (v1+vis)/(1.+p0[0])
    else:
        v1 = np.pi*p0[3]*mas2rad*np.sqrt(data['uu_smear']**2+data['vv_smear']**2)
        v1 += 1e-6*(v1 == 0)
        v1 = 2.*j1(v1)/v1
        v2 = 1.0+0.0j
        vis = v2*p0[0]*np.exp(-2.*np.pi*1.0j*mas2rad*(data['uu_smear']*p0[1]+data['vv_smear']*p0[2]))
        vis = (v1+vis)/(1.+p0[0])
        
        vis = vis.reshape((vis.shape[0], vis.shape[1]//smear, smear))
        vis = np.mean(vis, axis=2)
    
    return vis

def chi2_ud_bin(p0,
                data_list,
                observables,
                cov=False,
                smear=None):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
        p0[3]: float
            Uniform disk diameter (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    chi2: array
        Chi-squared of uniform disk with unresolved companion model.
    """
    
    chi2 = []
    for i in range(len(data_list)):
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
        p0_temp = np.array([p0[0].copy(), dra_temp, ddec_temp, p0[3].copy()])
        
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data_list[i],
                             smear=smear)
        sig = []
        err = []
        mod = []
        for j in range(len(observables)):
            if (observables[j] == 'v2'):
                sig += [data_list[i]['v2']]
                err += [data_list[i]['dv2']]
                mod += [v2v2(vis_mod,
                                 data=data_list[i])]
            elif (observables[j] == 'cp'):
                sig += [data_list[i]['cp']]
                err += [data_list[i]['dcp']]
                mod += [v2cp(vis_mod,
                               data=data_list[i])]
            elif (observables[j] == 'kp'):
                import pdb; pdb.set_trace()
        sig = np.concatenate(sig).flatten()
        mod = np.concatenate(mod).flatten()
        res = sig-mod
        if (cov == False):
            var = np.concatenate(err).flatten()**2
            res_icv = np.divide(res, var)
        else:
            if (data_list[i]['covflag'] == False):
                var = np.concatenate(err).flatten()**2
                res_icv = np.divide(res, var)
            else:
                res_icv = res.dot(data_list[i]['icv'])
        chi2 += [res_icv.dot(res)]
    
    return np.sum(chi2)

def lnprob_ud_bin(p0,
                  data_list,
                  observables,
                  cov=False,
                  smear=None,
                  temp=1.):
    """
    Parameters
    ----------
    p0: array
        p0[0]: float
            Relative flux of companion.
        p0[1]: float
            Right ascension offset of companion (mas).
        p0[2]: float
            Declination offset of companion (mas).
        p0[3]: float
            Uniform disk diameter (mas).
    data_list: list of dict
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    temp: float
        Covariance inflation factor.
    
    Returns
    -------
    lnprob: float
        Log-likelihood of uniform disk with unresolved companion model.
    """
    
    if ((p0[0] < 0.) or (np.abs(p0[1]) > 10000.) or (np.abs(p0[2]) > 10000.) or (p0[3] < 0.)):
        
        return -np.inf
    
    else:
        chi2 = chi2_ud_bin(p0,
                           data_list,
                           observables,
                           cov=cov,
                           smear=smear)
        
        return -0.5*np.sum(chi2)/temp

def chi2_ud_bin_fitdiamonly(theta0,
                            p0,
                            data_list,
                            observables,
                            cov=False,
                            smear=None):
    """
    Parameters
    ----------
    theta0: array
        theta0[0]: float
            Uniform disk diameter (mas).
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
        List of data whose chi-squared shall be computed. The list contains one
        data structure for each observation.
    observables: list of str
        List of observables which shall be considered.
    cov: bool
        True if covariance shall be considered.
    smear: int
        Numerical bandwidth smearing which shall be used.
    
    Returns
    -------
    chi2: array
        Chi-squared of uniform disk with unresolved companion model.
    """
    
    chi2 = []
    for i in range(len(data_list)):
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
        p0_temp = np.array([p0[0].copy(), dra_temp, ddec_temp, theta0[0].copy()])
        
        vis_mod = vis_ud_bin(p0=p0_temp,
                             data=data_list[i],
                             smear=smear)
        sig = []
        err = []
        mod = []
        for j in range(len(observables)):
            if (observables[j] == 'v2'):
                sig += [data_list[i]['v2']]
                err += [data_list[i]['dv2']]
                mod += [v2v2(vis_mod,
                                 data=data_list[i])]
            elif (observables[j] == 'cp'):
                sig += [data_list[i]['cp']]
                err += [data_list[i]['dcp']]
                mod += [v2cp(vis_mod,
                               data=data_list[i])]
            elif (observables[j] == 'kp'):
                import pdb; pdb.set_trace()
        sig = np.concatenate(sig).flatten()
        mod = np.concatenate(mod).flatten()
        res = sig-mod
        if (cov == False):
            var = np.concatenate(err).flatten()**2
            res_icv = np.divide(res, var)
        else:
            if (data_list[i]['covflag'] == False):
                var = np.concatenate(err).flatten()**2
                res_icv = np.divide(res, var)
            else:
                res_icv = res.dot(data_list[i]['icv'])
        chi2 += [res_icv.dot(res)]
    
    return np.sum(chi2)

def nsigma(chi2r_test,
           chi2r_true,
           ndof,
           use_mpmath=False):
    """
    Function for calculating the confidence level as
    defined in Eq. 1 of Absil et al. (2011).

    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.
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
    nsigma: float
        Detection significance.
    log_bin_prob: float
        Log-probability of a binary detection.
    """

    num_limit = False

    if not use_mpmath:
        bin_prob = stats.chi2.cdf(ndof*chi2r_test/chi2r_true, ndof)
        log_bin_prob = np.log10(bin_prob)

    else:
        # Decimal digits of precision
        mpmath.mp.dps = 50

        def chi2_cdf(x, k): 
            x, k = mpmath.mpf(x), mpmath.mpf(k) 
            return mpmath.gammainc(k/2, 0, x/2, regularized=True)

        bin_prob = chi2_cdf(ndof*chi2r_test/chi2r_true, float(ndof))
        log_bin_prob = float(mpmath.log10(bin_prob))
        bin_prob = float(bin_prob)

    nsigma = np.sqrt(stats.chi2.ppf(bin_prob, 1.))
    if (bin_prob > 1.-1e-15):
        nsigma = np.sqrt(stats.chi2.ppf(1.-1e-15, 1.))

    return nsigma, log_bin_prob

    # THIS IS WRONG (CF. CANDID)
    # p = stats.chi2.cdf(ndof, ndof*chi2r_test/chi2r_true)
    # log10p = np.log10(max(p, 10**(-155.623))) # 50 sigma max.
    # nsigma = np.sqrt(stats.chi2.ppf(1.-p, 1.))    
    # c = np.array([-0.25028407, 9.66640457]) # old
    # c = np.array([-0.29842513, 3.55829518]) # new
    # if (log10p < -15.):
    #     nsigma = np.polyval(c, log10p)
    # if (np.isnan(nsigma)):
    #     nsigma = 50.
    # return nsigma
