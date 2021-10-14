from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import block_diag


# =============================================================================
# MAIN
# =============================================================================

def invert(M):
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

def open(idir,
         fitsfile,
         verbose=True):
    """
    Parameters
    ----------
    idir: str
        Input directory where fits files are located.
    fitsfile: str
        Fits file which shall be opened.
    verbose: bool
        True if feedback shall be printed.
    
    Returns
    -------
    inst_list: list of str
        List of instruments from which data was opened.
    data_list: list of list of dict
        List of list of data which was opened. The list contains one list for
        each instrument, and this list contains one data structure for each
        observation.
    """
    
    hdul = pyfits.open(idir+fitsfile, memmap=False)
    try:
        hdul.index_of('OI_TARGET')
        inst_list, data_list = open_oifile(hdul)
    except:
        hdul.index_of('KP-DATA')
        try:
            hdul[0].header['PROCSOFT']
            inst_list, data_list = open_kpfile_new(hdul)
        except:
            inst_list, data_list = open_kpfile_old(hdul)
    hdul.close()
    
    if (verbose == True):
        for i in range(len(inst_list)):
            print('Opened '+inst_list[i]+' data')
            print('   %.0f observations' % len(data_list[i]))
            try:
                print('   %.0f baselines' % data_list[i][0]['vis2'].shape[0])
            except:
                None
            try:
                print('   %.0f triangles' % data_list[i][0]['t3'].shape[0])
            except:
                None
            try:
                print('   %.0f Fourier phases' % data_list[i][0]['kpmat'].shape[1])
                print('   %.0f kernel phases' % data_list[i][0]['kpmat'].shape[0])
            except:
                None
            print('   %.0f wavelengths' % data_list[i][0]['wave'].shape[0])
    
    return inst_list, data_list

def open_oifile(hdul):
    """
    Parameters
    ----------
    hdul: HDUList
        Fits file which shall be opened.
    
    Returns
    -------
    inst_list: list of str
        List of instruments from which data was opened.
    data_list: list of list of dict
        List of list of data which was opened. The list contains one list for
        each instrument, and this list contains one data structure for each
        observation.
    """
    
    data = {}
    for i in range(len(hdul)):
        try:
            if (hdul[i].header['EXTNAME'] == 'OI_WAVELENGTH'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['wave'] = np.append(data[inst]['wave'], hdul[i].data['EFF_WAVE'], axis=0)
                    data[inst]['dwave'] = np.append(data[inst]['dwave'], hdul[i].data['EFF_BAND'], axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['wave'] = hdul[i].data['EFF_WAVE']
                    data[inst]['dwave'] = hdul[i].data['EFF_BAND']
            if (hdul[i].header['EXTNAME'] == 'OI_VIS2'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['vis2'] = np.append(data[inst]['vis2'], hdul[i].data['VIS2DATA'], axis=0)
                    data[inst]['dvis2'] = np.append(data[inst]['dvis2'], hdul[i].data['VIS2ERR'], axis=0)
                    data[inst]['vis2u'] = np.append(data[inst]['vis2u'], hdul[i].data['UCOORD'], axis=0)
                    data[inst]['vis2v'] = np.append(data[inst]['vis2v'], hdul[i].data['VCOORD'], axis=0)
                    data[inst]['vis2sta'] = np.append(data[inst]['vis2sta'], hdul[i].data['STA_INDEX'], axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['vis2'] = hdul[i].data['VIS2DATA']
                    data[inst]['dvis2'] = hdul[i].data['VIS2ERR']
                    data[inst]['vis2u'] = hdul[i].data['UCOORD']
                    data[inst]['vis2v'] = hdul[i].data['VCOORD']
                    data[inst]['vis2sta'] = hdul[i].data['STA_INDEX']
            if (hdul[i].header['EXTNAME'] == 'OI_T3'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['t3'] = np.append(data[inst]['t3'], np.deg2rad(hdul[i].data['T3PHI']), axis=0)
                    data[inst]['dt3'] = np.append(data[inst]['dt3'], np.deg2rad(hdul[i].data['T3PHIERR']), axis=0)
                    data[inst]['t3sta'] = np.append(data[inst]['t3sta'], hdul[i].data['STA_INDEX'], axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['t3'] = np.deg2rad(hdul[i].data['T3PHI'])
                    data[inst]['dt3'] = np.deg2rad(hdul[i].data['T3PHIERR'])
                    data[inst]['t3sta'] = hdul[i].data['STA_INDEX']
            if (hdul[i].header['EXTNAME'] == 'VIS2COV'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['vis2cov'] = np.append(data[inst]['vis2cov'], hdul[i].data, axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['vis2cov'] = hdul[i].data
            if (hdul[i].header['EXTNAME'] == 'T3COV'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['t3cov'] = np.append(data[inst]['t3cov'], hdul[i].data, axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['t3cov'] = hdul[i].data
        except:
            continue
    
    inst_list = []
    data_list = []
    for i, key in enumerate(data.keys()):
        data[key]['base'] = np.sqrt(data[key]['vis2u']**2+data[key]['vis2v']**2)
        data[key]['uu'] = np.divide(data[key]['vis2u'][:, np.newaxis], data[key]['wave'][np.newaxis, :])
        data[key]['vv'] = np.divide(data[key]['vis2v'][:, np.newaxis], data[key]['wave'][np.newaxis, :])
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['vis2'].shape) == 1):
            data[key]['vis2'] = data[key]['vis2'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['dvis2'].shape) == 1):
            data[key]['dvis2'] = data[key]['dvis2'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['t3'].shape) == 1):
            data[key]['t3'] = data[key]['t3'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['dt3'].shape) == 1):
            data[key]['dt3'] = data[key]['dt3'][:, np.newaxis]
        nbase = np.unique(data[key]['vis2sta'], axis=0).shape[0]
        ntria = np.unique(data[key]['t3sta'], axis=0).shape[0]
        nobs1 = data[key]['vis2'].shape[0]//nbase
        nobs2 = data[key]['t3'].shape[0]//ntria
        if (nobs1 == nobs2):
            nobs = nobs1
        else:
            raise UserWarning('Number of squared visibility amplitudes does not match number of closure phases')
        inst_list += [key]
        data_list += [[]]
        for j in range(nobs):
            data_list[i] += [{}]
            data_list[i][j]['wave'] = data[key]['wave'].copy()
            data_list[i][j]['dwave'] = data[key]['dwave'].copy()
            data_list[i][j]['pa'] = 0.
            data_list[i][j]['vis2'] = data[key]['vis2'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['dvis2'] = data[key]['dvis2'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['vis2u'] = data[key]['vis2u'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['vis2v'] = data[key]['vis2v'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['base'] = data[key]['base'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['uu'] = data[key]['uu'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['vv'] = data[key]['vv'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['vis2sta'] = data[key]['vis2sta'][j*nbase:(j+1)*nbase].copy()
            data_list[i][j]['t3'] = data[key]['t3'][j*ntria:(j+1)*ntria].copy()
            data_list[i][j]['dt3'] = data[key]['dt3'][j*ntria:(j+1)*ntria].copy()
            data_list[i][j]['t3sta'] = data[key]['t3sta'][j*ntria:(j+1)*ntria].copy()
            
            covs = []
            try:
                covs += [data[key]['vis2cov'][j]]
            except:
                pass
            try:
                covs += [data[key]['t3cov'][j]]
            except:
                pass
            if (len(covs) > 0):
                data_list[i][j]['cov'] = block_diag(*covs)
                data_list[i][j]['icv'] = invert(data_list[i][j]['cov'])
                data_list[i][j]['covflag'] = True
            else:
                data_list[i][j]['covflag'] = False
            t3mat = np.zeros((data_list[i][j]['t3'].shape[0], data_list[i][j]['vis2'].shape[0]))
            for k in range(t3mat.shape[0]):
                base1 = data_list[i][j]['t3sta'][k][[0, 1]]
                base2 = data_list[i][j]['t3sta'][k][[1, 2]]
                base3 = data_list[i][j]['t3sta'][k][[2, 0]]
                flag1 = False
                flag2 = False
                flag3 = False
                l = 0
                while ((flag1 & flag2 & flag3) == False):
                    base = data_list[i][j]['vis2sta'][l]
                    if ((flag1 == False) & np.array_equal(base1, base)):
                        t3mat[k, l] = 1
                        flag1 = True
                    elif ((flag2 == False) & np.array_equal(base2, base)):
                        t3mat[k, l] = 1
                        flag2 = True
                    elif ((flag3 == False) & np.array_equal(base3, base)):
                        t3mat[k, l] = 1
                        flag3 = True
                    elif ((flag1 == False) & np.array_equal(base1[::-1], base)):
                        t3mat[k, l] = -1
                        flag1 = True
                    elif ((flag2 == False) & np.array_equal(base2[::-1], base)):
                        t3mat[k, l] = -1
                        flag2 = True
                    elif ((flag3 == False) & np.array_equal(base3[::-1], base)):
                        t3mat[k, l] = -1
                        flag3 = True
                    l += 1
            data_list[i][j]['t3mat'] = t3mat
            try:
                if (hdul[0].header['TELESCOP'] == 'ESO-VLTI-U1234'):
                    data_list[i][j]['diam'] = 8.2
                elif (hdul[0].header['TELESCOP'] == 'ESO-VLTI-A1234'):
                    data_list[i][j]['diam'] = 1.8
                elif (hdul[0].header['TELESCOP'] == 'JWST'):
                    data_list[i][j]['diam'] = 6.5
                else:
                    raise UserWarning('Telescope not known')
            except:
                if ('GRAVITY' in inst_list[i]):
                    data_list[i][j]['diam'] = 8.2
                elif ('PIONIER' in inst_list[i]):
                    data_list[i][j]['diam'] = 1.8
                else:
                    raise UserWarning('Telescope not known')
    
    return inst_list, data_list

def open_kpfile_old(hdul):
    """
    Parameters
    ----------
    hdul: HDUList
        Fits file which shall be opened.
    
    Returns
    -------
    inst_list: list of str
        List of instruments from which data was opened.
    data_list: list of list of dict
        List of list of data which was opened. The list contains one list for
        each instrument, and this list contains one data structure for each
        observation.
    """
    
    if (len(hdul['KP-DATA'].data.shape) == 1):
        try:
            inst_list = [hdul[0].header['INSTRUME']]
        except:
            inst_list = [hdul[0].header['CURRINST']]
        data_list = [[{}]]
        try:
            data_list[0][0]['wave'] = np.array([hdul[0].header['HIERARCH ESO INS CWLEN']*1e-6])
        except:
            data_list[0][0]['wave'] = np.array([hdul[0].header['CWAVEL']])
        data_list[0][0]['dwave'] = np.array([0.])
        try:
            data_list[0][0]['pa'] = np.mean(hdul['TEL'].data['DETPA'])
        except:
            data_list[0][0]['pa'] = np.mean(hdul['TEL'].data['pa'])
        data_list[0][0]['kp'] = hdul['KP-DATA'].data[:, np.newaxis]
        data_list[0][0]['dkp'] = np.sqrt(np.diag(hdul['KP-SIGM'].data))[:, np.newaxis]
        data_list[0][0]['kpu'] = -hdul['UV-PLANE'].data['UUC']
        data_list[0][0]['kpv'] = hdul['UV-PLANE'].data['VVC']
        data_list[0][0]['base'] = np.sqrt(data_list[0][0]['kpu']**2+data_list[0][0]['kpv']**2)
        data_list[0][0]['uu'] = np.divide(data_list[0][0]['kpu'][:, np.newaxis], data_list[0][0]['wave'][np.newaxis, :])
        data_list[0][0]['vv'] = np.divide(data_list[0][0]['kpv'][:, np.newaxis], data_list[0][0]['wave'][np.newaxis, :])
        try:
            data_list[0][0]['cov'] = hdul['KP-SIGM'].data
            data_list[0][0]['icv'] = invert(data_list[0][0]['cov'])
            data_list[0][0]['covflag'] = True
        except:
            data_list[0][0]['covflag'] = False
        data_list[0][0]['kpmat'] = hdul['KER-MAT'].data    
        if ('ESO-VLT' in hdul[0].header['TELESCOP']):
            data_list[0][0]['diam'] = 8.2
        elif ('Keck' in hdul[0].header['TELESCOP']):
            data_list[0][0]['diam'] = 10.95
        else:
            raise UserWarning('Telescope not known')
    elif (len(hdul['KP-DATA'].data.shape) == 2):
        nobs = hdul['KP-DATA'].data.shape[0]
        try:
            inst_list = [hdul[0].header['INSTRUME']]
        except:
            inst_list = [hdul[0].header['CURRINST']]
        data_list = []
        for i in range(nobs):
            temp = {}
            try:
                temp['wave'] = np.array([hdul[0].header['HIERARCH ESO INS CWLEN']*1e-6])
            except:
                temp['wave'] = np.array([hdul[0].header['CWAVEL']])
            temp['dwave'] = np.array([0.])
            try:
                temp['pa'] = hdul['TEL'].data['DETPA'].copy()[i]
            except:
                temp['pa'] = hdul['TEL'].data['pa'].copy()[i]
            temp['kp'] = hdul['KP-DATA'].data.copy()[i, :, np.newaxis]
            temp['dkp'] = np.sqrt(np.diag(hdul['KP-SIGM'].data.copy()[i]))[:, np.newaxis]
            temp['kpu'] = -hdul['UV-PLANE'].data['UUC'].copy()
            temp['kpv'] = hdul['UV-PLANE'].data['VVC'].copy()
            temp['base'] = np.sqrt(temp['kpu']**2+temp['kpv']**2)
            temp['uu'] = np.divide(temp['kpu'][:, np.newaxis], temp['wave'][np.newaxis, :])
            temp['vv'] = np.divide(temp['kpv'][:, np.newaxis], temp['wave'][np.newaxis, :])
            try:
                temp['cov'] = hdul['KP-SIGM'].data.copy()[i]
                temp['icv'] = invert(temp['cov'])
                temp['covflag'] = True
            except:
                temp['covflag'] = False
            temp['kpmat'] = hdul['KER-MAT'].data.copy()
            if ('ESO-VLT' in hdul[0].header['TELESCOP']):
                temp['diam'] = 8.2
            elif ('Keck' in hdul[0].header['TELESCOP']):
                temp['diam'] = 10.95
            else:
                raise UserWarning('Telescope not known')
            data_list += [temp]
        data_list = [data_list]
    
    return inst_list, data_list

def open_kpfile_new(hdul):
    """
    Parameters
    ----------
    hdul: HDUList
        Fits file which shall be opened.
    
    Returns
    -------
    inst_list: list of str
        List of instruments from which data was opened.
    data_list: list of list of dict
        List of list of data which was opened. The list contains one list for
        each instrument, and this list contains one data structure for each
        observation.
    """
    
    nobs = hdul['KP-DATA'].data.shape[0]
    inst_list = [hdul[0].header['INSTRUME']]
    data_list = []
    for i in range(nobs):
        temp = {}
        temp['wave'] = hdul['CWAVEL'].data['CWAVEL']
        temp['dwave'] = hdul['CWAVEL'].data['DWAVEL']
        temp['pa'] = hdul['DETPA'].data[i]
        temp['kp'] = np.swapaxes(hdul['KP-DATA'].data.copy()[i], 0, 1)
        # temp['dkp'] = np.swapaxes(hdul['KP-SIGM'].data.copy()[i], 0, 1)
        temp['dkp'] = np.swapaxes(hdul['EKP-SIGM'].data.copy()[i], 0, 1)
        temp['kpu'] = -hdul['UV-PLANE'].data['UUC'].copy()
        temp['kpv'] = hdul['UV-PLANE'].data['VVC'].copy()
        temp['base'] = np.sqrt(temp['kpu']**2+temp['kpv']**2)
        temp['uu'] = np.divide(temp['kpu'][:, np.newaxis], temp['wave'][np.newaxis, :])
        temp['vv'] = np.divide(temp['kpv'][:, np.newaxis], temp['wave'][np.newaxis, :])
        try:
            # temp['cov'] = hdul['KP-COV'].data.copy()[i, 0] # FIXME
            temp['cov'] = hdul['EKP-COV'].data.copy()[i, 0] # FIXME
            temp['icv'] = invert(temp['cov'])
            temp['covflag'] = True
        except:
            temp['covflag'] = False
        temp['kpmat'] = hdul['KER-MAT'].data.copy()
        temp['diam'] = hdul[0].header['DIAM']
        data_list += [temp]
    data_list = [data_list]
    
    return inst_list, data_list
