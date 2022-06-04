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
    if ('OI_TARGET' in hdul):
        inst_list, data_list = open_oifile(hdul)
    elif ('KP-DATA' in hdul):
        if ('PROCSOFT' in hdul[0].header):
            inst_list, data_list = open_kpfile_new(hdul)
        else:
            inst_list, data_list = open_kpfile_old(hdul)
    else:
        raise UserWarning('Unknown file type')
    hdul.close()
    
    if (verbose == True):
        for i in range(len(inst_list)):
            print('Opened '+inst_list[i]+' data')
            print('   %.0f observations' % len(data_list[i]))
            try:
                print('   %.0f baselines' % data_list[i][0]['v2'].shape[0])
            except:
                None
            try:
                print('   %.0f triangles' % data_list[i][0]['cp'].shape[0])
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
    klflag = False
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
                    if ((klflag == True) or ('VIS2DATA' in hdul)):
                        klflag = True
                        data[inst]['v2'] = np.append(data[inst]['v2'], hdul['VIS2DATA'].data, axis=0)
                        data[inst]['dv2'] = np.append(data[inst]['dv2'], hdul['VIS2ERR'].data, axis=0)
                    else:
                        data[inst]['v2'] = np.append(data[inst]['v2'], hdul[i].data['VIS2DATA'], axis=0)
                        data[inst]['dv2'] = np.append(data[inst]['dv2'], hdul[i].data['VIS2ERR'], axis=0)
                    data[inst]['v2u'] = np.append(data[inst]['v2u'], hdul[i].data['UCOORD'], axis=0)
                    data[inst]['v2v'] = np.append(data[inst]['v2v'], hdul[i].data['VCOORD'], axis=0)
                    data[inst]['v2sta'] = np.append(data[inst]['v2sta'], hdul[i].data['STA_INDEX'], axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    if ((klflag == True) or ('VIS2DATA' in hdul)):
                        klflag = True
                        data[inst]['v2'] = hdul['VIS2DATA'].data
                        data[inst]['dv2'] = hdul['VIS2ERR'].data
                    else:
                        data[inst]['v2'] = hdul[i].data['VIS2DATA']
                        data[inst]['dv2'] = hdul[i].data['VIS2ERR']
                    data[inst]['v2u'] = hdul[i].data['UCOORD']
                    data[inst]['v2v'] = hdul[i].data['VCOORD']
                    data[inst]['v2sta'] = hdul[i].data['STA_INDEX']
            if (hdul[i].header['EXTNAME'] == 'OI_T3'):
                inst = hdul[i].header['INSNAME']
                try:
                    if ((klflag == True) or ('T3PHI' in hdul)):
                        klflag = True
                        data[inst]['cp'] = np.append(data[inst]['cp'], np.deg2rad(hdul['T3PHI'].data), axis=0)
                        data[inst]['dcp'] = np.append(data[inst]['dcp'], np.deg2rad(hdul['T3PHIERR'].data), axis=0)
                    else:
                        data[inst]['cp'] = np.append(data[inst]['cp'], np.deg2rad(hdul[i].data['T3PHI']), axis=0)
                        data[inst]['dcp'] = np.append(data[inst]['dcp'], np.deg2rad(hdul[i].data['T3PHIERR']), axis=0)
                    data[inst]['cpsta'] = np.append(data[inst]['cpsta'], hdul[i].data['STA_INDEX'], axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    if ((klflag == True) or ('T3PHI' in hdul)):
                        klflag = True
                        data[inst]['cp'] = np.deg2rad(hdul['T3PHI'].data)
                        data[inst]['dcp'] = np.deg2rad(hdul['T3PHIERR'].data)
                    else:
                        data[inst]['cp'] = np.deg2rad(hdul[i].data['T3PHI'])
                        data[inst]['dcp'] = np.deg2rad(hdul[i].data['T3PHIERR'])
                    data[inst]['cpsta'] = hdul[i].data['STA_INDEX']
            if (hdul[i].header['EXTNAME'] == 'V2COV'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['v2cov'] = np.append(data[inst]['v2cov'], hdul[i].data, axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['v2cov'] = hdul[i].data
            if (hdul[i].header['EXTNAME'] == 'CPCOV'):
                inst = hdul[i].header['INSNAME']
                try:
                    data[inst]['cpcov'] = np.append(data[inst]['cpcov'], hdul[i].data, axis=0)
                except:
                    if (inst not in data):
                        data[inst] = {}
                    data[inst]['cpcov'] = hdul[i].data
        except:
            continue
    
    inst_list = []
    data_list = []
    for i, key in enumerate(data.keys()):
        data[key]['base'] = np.sqrt(data[key]['v2u']**2+data[key]['v2v']**2)
        data[key]['uu'] = np.divide(data[key]['v2u'][:, np.newaxis], data[key]['wave'][np.newaxis, :])
        data[key]['vv'] = np.divide(data[key]['v2v'][:, np.newaxis], data[key]['wave'][np.newaxis, :])
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['v2'].shape) == 1):
            data[key]['v2'] = data[key]['v2'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['dv2'].shape) == 1):
            data[key]['dv2'] = data[key]['dv2'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['cp'].shape) == 1):
            data[key]['cp'] = data[key]['cp'][:, np.newaxis]
        if (len(data[key]['uu'].shape) == 2) and (len(data[key]['dcp'].shape) == 1):
            data[key]['dcp'] = data[key]['dcp'][:, np.newaxis]
        nbase = np.unique(data[key]['v2sta'], axis=0).shape[0]
        ntria = np.unique(data[key]['cpsta'], axis=0).shape[0]
        if (klflag == True):
            nobs = 1
        else:
            nobs1 = data[key]['v2'].shape[0]//nbase
            nobs2 = data[key]['cp'].shape[0]//ntria
            if (nobs1 == nobs2):
                nobs = nobs1
            else:
                raise UserWarning('Number of squared visibility amplitudes does not match number of closure phases')
        inst_list += [key]
        data_list += [[]]
        for j in range(nobs):
            if (klflag == True):
                data_list[i] += [{}]
                data_list[i][j]['wave'] = data[key]['wave'].copy()
                data_list[i][j]['dwave'] = data[key]['dwave'].copy()
                data_list[i][j]['pa'] = 0.
                data_list[i][j]['v2'] = data[key]['v2'].copy()
                data_list[i][j]['dv2'] = data[key]['dv2'].copy()
                data_list[i][j]['v2u'] = data[key]['v2u'].copy()
                data_list[i][j]['v2v'] = data[key]['v2v'].copy()
                data_list[i][j]['base'] = data[key]['base'].copy()
                data_list[i][j]['uu'] = data[key]['uu'].copy()
                data_list[i][j]['vv'] = data[key]['vv'].copy()
                data_list[i][j]['v2sta'] = data[key]['v2sta'].copy()
                data_list[i][j]['cp'] = data[key]['cp'].copy()
                data_list[i][j]['dcp'] = data[key]['dcp'].copy()
                data_list[i][j]['cpsta'] = data[key]['cpsta'].copy()
            else:
                data_list[i] += [{}]
                data_list[i][j]['wave'] = data[key]['wave'].copy()
                data_list[i][j]['dwave'] = data[key]['dwave'].copy()
                data_list[i][j]['pa'] = 0.
                data_list[i][j]['v2'] = data[key]['v2'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['dv2'] = data[key]['dv2'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['v2u'] = data[key]['v2u'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['v2v'] = data[key]['v2v'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['base'] = data[key]['base'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['uu'] = data[key]['uu'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['vv'] = data[key]['vv'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['v2sta'] = data[key]['v2sta'][j*nbase:(j+1)*nbase].copy()
                data_list[i][j]['cp'] = data[key]['cp'][j*ntria:(j+1)*ntria].copy()
                data_list[i][j]['dcp'] = data[key]['dcp'][j*ntria:(j+1)*ntria].copy()
                data_list[i][j]['cpsta'] = data[key]['cpsta'][j*ntria:(j+1)*ntria].copy()
            try:
                data_list[i][j]['v2cov'] = data[key]['v2cov'][j]
            except:
                pass
            try:
                data_list[i][j]['cpcov'] = data[key]['cpcov'][j]
            except:
                pass
            cpmat = np.zeros((data_list[i][j]['cpsta'].shape[0], data_list[i][j]['v2sta'].shape[0]))
            for k in range(cpmat.shape[0]):
                base1 = data_list[i][j]['cpsta'][k][[0, 1]]
                base2 = data_list[i][j]['cpsta'][k][[1, 2]]
                base3 = data_list[i][j]['cpsta'][k][[2, 0]]
                flag1 = False
                flag2 = False
                flag3 = False
                l = 0
                while ((flag1 & flag2 & flag3) == False):
                    base = data_list[i][j]['v2sta'][l]
                    if ((flag1 == False) & np.array_equal(base1, base)):
                        cpmat[k, l] = 1
                        flag1 = True
                    elif ((flag2 == False) & np.array_equal(base2, base)):
                        cpmat[k, l] = 1
                        flag2 = True
                    elif ((flag3 == False) & np.array_equal(base3, base)):
                        cpmat[k, l] = 1
                        flag3 = True
                    elif ((flag1 == False) & np.array_equal(base1[::-1], base)):
                        cpmat[k, l] = -1
                        flag1 = True
                    elif ((flag2 == False) & np.array_equal(base2[::-1], base)):
                        cpmat[k, l] = -1
                        flag2 = True
                    elif ((flag3 == False) & np.array_equal(base3[::-1], base)):
                        cpmat[k, l] = -1
                        flag3 = True
                    l += 1
            data_list[i][j]['cpmat'] = cpmat
            if (klflag == True):
                data_list[i][j]['v2mat'] = hdul['V2PROJ'].data
                data_list[i][j]['cpmat'] = np.dot(hdul['CPPROJ'].data, data_list[i][j]['cpmat'])
                data_list[i][j]['klflag'] = True
            else:
                data_list[i][j]['klflag'] = False
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
                elif ('SPHERE' in inst_list[i]):
                    data_list[i][j]['diam'] = 8.2
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
            data_list[0][0]['kpcov'] = hdul['KP-SIGM'].data
        except:
            pass
        data_list[0][0]['klflag'] = False # only relevant for OIFITS files
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
                temp['kpcov'] = hdul['KP-SIGM'].data.copy()[i]
            except:
                pass
            temp['klflag'] = False # only relevant for OIFITS files
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
    ekp = True
    for i in range(nobs):
        temp = {}
        temp['wave'] = hdul['CWAVEL'].data['CWAVEL']
        temp['dwave'] = hdul['CWAVEL'].data['DWAVEL']
        temp['pa'] = hdul['DETPA'].data[i]
        temp['kp'] = np.swapaxes(hdul['KP-DATA'].data.copy()[i], 0, 1)
        if (ekp == True):
            try:
                temp['dkp'] = np.swapaxes(hdul['EKP-SIGM'].data.copy()[i], 0, 1)
            except:
                ekp = False
        if (ekp == False):
            temp['dkp'] = np.swapaxes(hdul['KP-SIGM'].data.copy()[i], 0, 1)
        temp['kpu'] = -hdul['UV-PLANE'].data['UUC'].copy()
        temp['kpv'] = hdul['UV-PLANE'].data['VVC'].copy()
        temp['base'] = np.sqrt(temp['kpu']**2+temp['kpv']**2)
        temp['uu'] = np.divide(temp['kpu'][:, np.newaxis], temp['wave'][np.newaxis, :])
        temp['vv'] = np.divide(temp['kpv'][:, np.newaxis], temp['wave'][np.newaxis, :])
        try:
            if (ekp == True):
                temp['kpcov'] = hdul['EKP-COV'].data.copy()[i, 0]
            else:
                temp['kpcov'] = hdul['KP-COV'].data.copy()[i, 0]
        except:
            pass
        temp['klflag'] = False # only relevant for OIFITS files
        temp['kpmat'] = hdul['KER-MAT'].data.copy()
        temp['diam'] = hdul[0].header['DIAM']
        data_list += [temp]
    data_list = [data_list]
    
    return inst_list, data_list
