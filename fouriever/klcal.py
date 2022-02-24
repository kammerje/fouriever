from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import glob
import os
import sys

from . import inst

observables_known = ['vis2', 't3', 'kp']


# =============================================================================
# MAIN
# =============================================================================

class data():
    
    def __init__(self,
                 scidir,
                 scifiles,
                 caldir,
                 calfiles):
        """
        Parameters
        ----------
        scidir: str
            Input directory where science fits files are located.
        scifiles: list of str, None
            List of science fits files which shall be opened. All fits files
            from ``scidir`` are opened with ``scifiles=None``.
        caldir: str
            Input directory where calibrator fits files are located.
        calfiles: list of str, None
            List of calibrator fits files which shall be opened. All fits
            files from ``caldir`` are opened with ``calfiles=None``.
        """
        
        self.scidir = scidir
        self.scifiles = scifiles
        self.caldir = caldir
        self.calfiles = calfiles
        
        if (self.scifiles is None):
            self.scifiles = glob.glob(self.scidir+'*fits')
            for i, item in enumerate(self.scifiles):
                head, tail = os.path.split(item)
                self.scifiles[i] = tail
        
        if (self.calfiles is None):
            self.calfiles = glob.glob(self.caldir+'*fits')
            for i, item in enumerate(self.calfiles):
                head, tail = os.path.split(item)
                self.calfiles[i] = tail
        
        self.sci_inst_list = []
        self.sci_data_list = []
        for i in range(len(self.scifiles)):
            inst_list, data_list = inst.open(idir=scidir,
                                             fitsfile=self.scifiles[i],
                                             verbose=False)
            self.sci_inst_list += inst_list
            self.sci_data_list += data_list
        
        self.cal_inst_list = []
        self.cal_data_list = []
        for i in range(len(self.calfiles)):
            inst_list, data_list = inst.open(idir=caldir,
                                             fitsfile=self.calfiles[i],
                                             verbose=False)
            self.cal_inst_list += inst_list
            self.cal_data_list += data_list
        
        self.set_inst(inst=self.sci_inst_list[0])
        self.set_observables(self.get_observables())
        
        return None
    
    def get_inst(self):
        """
        Returns
        -------
        inst_list: list of str
            List of instruments from which data was opened.
        """
        
        return self.sci_inst_list
    
    def set_inst(self,
                 inst):
        """
        Parameters
        ----------
        inst: str
            Instrument which shall be selected.
        """
        
        if (inst in self.sci_inst_list):
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
        ww = np.where(np.array(self.sci_inst_list) == self.inst)[0]
        for i in range(len(observables_known)):
            j = 0
            flag = True
            while (j < len(ww) and flag):
                if (observables_known[i] not in self.sci_data_list[ww[j]][0].keys()):
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
    
    def calibrate(self,
                  odir,
                  K_klip=4):
        """
        Parameters
        ----------
        odir: str
            Output directory where calibrated science fits files shall be saved to.
        K_klip: int
            Order of Karhunen-Loeve calibration.
        """
        
        self.decompose(K_klip=K_klip)
        self.project(odir=odir)
        
        return None
    
    def decompose(self,
                  K_klip=4):
        """
        Parameters
        ----------
        K_klip: int
            Order of Karhunen-Loeve calibration.
        """
        
        print('Computing Karhunen-Loeve decomposition')
        print('   K_klip = %.0f' % int(K_klip))
        
        data_list = []
        ww = np.where(np.array(self.cal_inst_list) == self.inst)[0]
        for i in range(len(ww)):
            for j in range(len(self.cal_data_list[ww[i]])):
                data_list += [self.cal_data_list[ww[i]][j]]
        
        self.P = {}
        for i in range(len(self.observables)):
            data_temp = []
            for j in range(len(data_list)):
                try:
                    data_temp += [data_list[j][self.observables[i]].flatten()]
                except:
                    pass
            
            print('   '+self.observables[i]+': '+str(len(data_temp))+' data sets')
            
            if (int(K_klip) > len(data_temp)):
                raise ValueError('K_klip cannot be larger than the number of calibrator data sets')
            data_temp = np.array(data_temp)
            
            E_RR = data_temp.dot(data_temp.T) # Equation 13 in Kammerer et al. 2019
            w, v = np.linalg.eigh(E_RR)
            v_sort = np.zeros(v.shape) # eigenvectors sorted in descending order
            temp = np.argsort(w)[::-1]
            for j in range(len(w)):
                v_sort[:, j] = v[:, temp[j]]
            w_sort = np.sort(w)[::-1] # eigenvalues sorted in descending order
            w_sort[w_sort <= 0.] = 1e-10
            
            v_norm = np.divide(v_sort, np.sqrt(w_sort))
            Z_KL = data_temp.T.dot(v_norm) # Equation 14 in Kammerer et al. 2019
            
            Z_prim = Z_KL[:, :int(K_klip)]
            P = np.identity(Z_prim.shape[0])-Z_prim.dot(Z_prim.T) # Equation 15 in Kammerer et al. 2019
            
            w, v = np.linalg.eigh(P)
            self.P[self.observables[i]] = v[np.where(w > 1e-3)[0]].dot(np.diag(w)).dot(v.T)
            
            print('   '+self.observables[i]+': projection matrix shape = '+str(self.P[self.observables[i]].shape))
        
        return None
    
    def project(self,
                odir):
        """
        Parameters
        ----------
        odir: str
            Output directory where calibrated science fits files shall be saved to.
        """
        
        print('Computing Karhunen-Loeve projection')
        
        if (odir == self.scidir):
            raise UserWarning('Using odir = scidir would overwrite the science data')
        if (not os.path.exists(odir)):
            os.makedirs(odir)
        
        Nscifiles = len(self.scifiles)
        for i in range(Nscifiles):
            hdul = pyfits.open(self.scidir+self.scifiles[i], memmap=False)
            
            if ('KP-DATA' in hdul):
                sys.stdout.write('\r   File %.0f of %.0f: kernel phase FITS file' % (i+1, Nscifiles))
                sys.stdout.flush()
                
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'kp'):
                        
                        hdul['KER-MAT'].data = self.P[self.observables[j]].dot(hdul['KER-MAT'].data)
                        
                        if (len(hdul['KP-DATA'].data.shape) == 1):
                            hdul['KP-DATA'].data = self.P[self.observables[j]].dot(hdul['KP-DATA'].data)
                        elif (len(hdul['KP-DATA'].data.shape) == 2):
                            temp = np.zeros((hdul['KP-DATA'].data.shape[0], self.P[self.observables[j]].shape[0]))
                            for k in range(hdul['KP-DATA'].data.shape[0]):
                                temp[k] = self.P[self.observables[j]].dot(hdul['KP-DATA'].data[k])
                            hdul['KP-DATA'].data = temp
                        elif (len(hdul['KP-DATA'].data.shape) == 3):
                            temp = np.zeros((hdul['KP-DATA'].data.shape[0], hdul['KP-DATA'].data.shape[1], self.P[self.observables[j]].shape[0]))
                            for k in range(hdul['KP-DATA'].data.shape[0]):
                                for l in range(hdul['KP-DATA'].data.shape[1]):
                                    temp[k, l] = self.P[self.observables[j]].dot(hdul['KP-DATA'].data[k, l])
                            hdul['KP-DATA'].data = temp
                        
                        if (len(hdul['KP-SIGM'].data.shape) == 1):
                            var = np.diag(hdul['KP-SIGM'].data**2)
                            cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                            hdul['KP-SIGM'].data = np.sqrt(np.diag(cov))
                        elif (len(hdul['KP-SIGM'].data.shape) == 2):
                            try:
                                hdul[0].header['PROCSOFT']
                                temp = np.zeros((hdul['KP-SIGM'].data.shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['KP-SIGM'].data.shape[0]):
                                    var = np.diag(hdul['KP-SIGM'].data[k]**2)
                                    cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                                    temp[k] = np.sqrt(np.diag(cov))
                                hdul['KP-SIGM'].data = temp
                            except:
                                hdul['KP-SIGM'].data = self.P[self.observables[j]].dot(hdul['KP-SIGM'].data).dot(self.P[self.observables[j]].T)
                        elif (len(hdul['KP-SIGM'].data.shape) == 3):
                            try:
                                hdul[0].header['PROCSOFT']
                                temp = np.zeros((hdul['KP-SIGM'].data.shape[0], hdul['KP-SIGM'].data.shape[1], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['KP-SIGM'].data.shape[0]):
                                    for l in range(hdul['KP-SIGM'].data.shape[1]):
                                        var = np.diag(hdul['KP-SIGM'].data[k, l]**2)
                                        cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                                        temp[k, l] = np.sqrt(np.diag(cov))
                                hdul['KP-SIGM'].data = temp
                            except:
                                temp = np.zeros((hdul['KP-SIGM'].data.shape[0], self.P[self.observables[j]].shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['KP-SIGM'].data.shape[0]):
                                    temp[k] = self.P[self.observables[j]].dot(hdul['KP-SIGM'].data[k]).dot(self.P[self.observables[j]].T)
                                hdul['KP-SIGM'].data = temp
                        
                        try:
                            if (len(hdul['EKP-SIGM'].data.shape) == 1):
                                var = np.diag(hdul['EKP-SIGM'].data**2)
                                cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                                hdul['EKP-SIGM'].data = np.sqrt(np.diag(cov))
                            elif (len(hdul['EKP-SIGM'].data.shape) == 2):
                                temp = np.zeros((hdul['EKP-SIGM'].data.shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['EKP-SIGM'].data.shape[0]):
                                    var = np.diag(hdul['EKP-SIGM'].data[k]**2)
                                    cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                                    temp[k] = np.sqrt(np.diag(cov))
                                hdul['EKP-SIGM'].data = temp
                            elif (len(hdul['EKP-SIGM'].data.shape) == 3):
                                temp = np.zeros((hdul['EKP-SIGM'].data.shape[0], hdul['EKP-SIGM'].data.shape[1], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['EKP-SIGM'].data.shape[0]):
                                    for l in range(hdul['EKP-SIGM'].data.shape[1]):
                                        var = np.diag(hdul['EKP-SIGM'].data[k, l]**2)
                                        cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                                        temp[k, l] = np.sqrt(np.diag(cov))
                                hdul['EKP-SIGM'].data = temp
                        except:
                            pass
                        
                        try:
                            if (len(hdul['KP-COV'].data.shape) == 2):
                                hdul['KP-COV'].data = self.P[self.observables[j]].dot(hdul['KP-COV'].data).dot(self.P[self.observables[j]].T)
                            elif (len(hdul['KP-COV'].data.shape) == 3):
                                temp = np.zeros((hdul['KP-COV'].data.shape[0], self.P[self.observables[j]].shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['KP-COV'].data.shape[0]):
                                    temp[k] = self.P[self.observables[j]].dot(hdul['KP-COV'].data[k]).dot(self.P[self.observables[j]].T)
                                hdul['KP-COV'].data = temp
                            elif (len(hdul['KP-COV'].data.shape) == 4):
                                temp = np.zeros((hdul['KP-COV'].data.shape[0], hdul['KP-COV'].data.shape[1], self.P[self.observables[j]].shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['KP-COV'].data.shape[0]):
                                    for l in range(hdul['KP-COV'].data.shape[1]):
                                        temp[k, l] = self.P[self.observables[j]].dot(hdul['KP-COV'].data[k, l]).dot(self.P[self.observables[j]].T)
                                hdul['KP-COV'].data = temp
                        except:
                            pass
                        
                        try:
                            if (len(hdul['EKP-COV'].data.shape) == 2):
                                hdul['EKP-COV'].data = self.P[self.observables[j]].dot(hdul['EKP-COV'].data).dot(self.P[self.observables[j]].T)
                            elif (len(hdul['EKP-COV'].data.shape) == 3):
                                temp = np.zeros((hdul['EKP-COV'].data.shape[0], self.P[self.observables[j]].shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['EKP-COV'].data.shape[0]):
                                    temp[k] = self.P[self.observables[j]].dot(hdul['EKP-COV'].data[k]).dot(self.P[self.observables[j]].T)
                                hdul['EKP-COV'].data = temp
                            elif (len(hdul['EKP-COV'].data.shape) == 4):
                                temp = np.zeros((hdul['EKP-COV'].data.shape[0], hdul['EKP-COV'].data.shape[1], self.P[self.observables[j]].shape[0], self.P[self.observables[j]].shape[0]))
                                for k in range(hdul['EKP-COV'].data.shape[0]):
                                    for l in range(hdul['EKP-COV'].data.shape[1]):
                                        temp[k, l] = self.P[self.observables[j]].dot(hdul['EKP-COV'].data[k, l]).dot(self.P[self.observables[j]].T)
                                hdul['EKP-COV'].data = temp
                        except:
                            pass
                
                hdul.writeto(odir+self.scifiles[i][:-5]+'_klcal.fits', overwrite=True, output_verify='fix')
                hdul.close()
            
            elif (('OI_VIS2' in hdul) and ('OI_T3' in hdul)):
                sys.stdout.write('\r   File %.0f of %.0f: OIFITS file' % (i+1, Nscifiles))
                sys.stdout.flush()
                
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'vis2'):
                        
                        hdu0 = pyfits.ImageHDU(self.P[self.observables[j]])
                        hdu0.header['EXTNAME'] = 'VIS2PROJ'
                        if (hdul['OI_VIS2'].data['VIS2DATA'].ndim == 1):
                            hdu1 = pyfits.ImageHDU(self.P[self.observables[j]].dot(hdul['OI_VIS2'].data['VIS2DATA']))
                            hdu1.header['EXTNAME'] = 'VIS2DATA'
                            var = np.diag(hdul['OI_VIS2'].data['VIS2ERR']**2)
                            cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                            hdu2 = pyfits.ImageHDU(np.sqrt(np.diag(cov)))
                            hdu2.header['EXTNAME'] = 'VIS2ERR'
                            if ('VIS2COV' in hdul):
                                hdul['VIS2COV'].data = self.P[self.observables[j]].dot(hdul['VIS2COV'].data[0]).dot(self.P[self.observables[j]].T)[np.newaxis, :]
                        else:
                            raise UserWarning('Only 1D is implemented for VIS2DATA')
                        hdul += [hdu0, hdu1, hdu2]
                    
                    elif (self.observables[j] == 't3'):
                        
                        hdu0 = pyfits.ImageHDU(self.P[self.observables[j]])
                        hdu0.header['EXTNAME'] = 'T3PROJ'
                        if (hdul['OI_T3'].data['T3PHI'].ndim == 1):
                            hdu1 = pyfits.ImageHDU(self.P[self.observables[j]].dot(hdul['OI_T3'].data['T3PHI']))
                            hdu1.header['EXTNAME'] = 'T3PHI'
                            var = np.diag(np.deg2rad(hdul['OI_T3'].data['T3PHIERR'])**2)
                            cov = self.P[self.observables[j]].dot(var).dot(self.P[self.observables[j]].T)
                            hdu2 = pyfits.ImageHDU(np.rad2deg(np.sqrt(np.diag(cov))))
                            hdu2.header['EXTNAME'] = 'T3PHIERR'
                            if ('T3COV' in hdul):
                                hdul['T3COV'].data = self.P[self.observables[j]].dot(hdul['T3COV'].data[0]).dot(self.P[self.observables[j]].T)[np.newaxis, :]
                        else:
                            raise UserWarning('Only 1D is implemented for T3PHI')
                        hdul += [hdu0, hdu1, hdu2]
                
                hdul.writeto(odir+self.scifiles[i][:-7]+'_klcal.oifits', overwrite=True, output_verify='fix')
                hdul.close()
            
            else:
                raise UserWarning('Support for this data format is not implemented')
        print('')
        
        return None
