from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from . import inst


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
            Input directory where the science fits files are located.
        scifiles: list of str
            List of science fits files which shall be opened.
        caldir: str
            Input directory where the calibrator fits files are located.
        calfiles: list of str
            List of calibrator fits files which shall be opened.
        """
        
        self.scidir = scidir
        self.scifiles = scifiles
        self.caldir = caldir
        self.calfiles = calfiles
        
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
            List of observables available for the currently selected instrument.
        """
        
        observables_known = ['vis', 'vis2', 't3', 'kp']
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
            Output directory where the calibrated science fits files shall be saved to.
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
            w_sort[w_sort <= 0.] = 1e-30
            
            v_norm = np.divide(v_sort, np.sqrt(w_sort))
            Z_KL = data_temp.T.dot(v_norm) # Equation 14 in Kammerer et al. 2019
            
            Z_prim = Z_KL[:, :int(K_klip)]
            P = np.identity(Z_prim.shape[0])-Z_prim.dot(Z_prim.T) # Equation 15 in Kammerer et al. 2019
            
            w, v = np.linalg.eigh(P)
            self.P[self.observables[i]] = v[np.where(w > 1e-10)[0]].dot(np.diag(w)).dot(v.T)
            
            print('   '+self.observables[i]+': projection matrix shape = '+str(self.P[self.observables[i]].shape))
        
        return None
    
    def project(self,
                odir):
        """
        Parameters
        ----------
        odir: str
            Output directory where the calibrated science fits files shall be saved to.
        """
        
        print('Computing Karhunen-Loeve projection')
        
        if (odir == self.scidir):
            raise UserWarning('Using odir = scidir would overwrite the science data')
        if (not os.path.exists(odir)):
            os.makedirs(odir)
        
        data_before = []
        data_after = []
        Nscifiles = len(self.scifiles)
        for i in range(Nscifiles):
            hdul = pyfits.open(self.scidir+self.scifiles[i])
            
            try:
                try:
                    inst = hdul[0].header['INSTRUME']
                except:
                    inst = hdul[0].header['CURRINST']
                sys.stdout.write('\r   File %.0f of %.0f: kernel phase FITS file' % (i+1, Nscifiles))
                sys.stdout.flush()
                
                for j in range(len(self.observables)):
                    if (self.observables[j] == 'kp'):
                        data_before += [hdul['KP-DATA'].data.flatten()]
                        hdul['KER-MAT'].data = self.P[self.observables[j]].dot(hdul['KER-MAT'].data)
                        hdul['KP-DATA'].data = self.P[self.observables[j]].dot(hdul['KP-DATA'].data)
                        hdul['KP-SIGM'].data = self.P[self.observables[j]].dot(hdul['KP-SIGM'].data).dot(self.P[self.observables[j]].T)
                        data_after += [hdul['KP-DATA'].data.flatten()]
            
            except:
                sys.stdout.write('\r   File %.0f of %.0f: OIFITS file' % (i+1, Nscifiles))
                sys.stdout.flush()
                
                raise UserWarning('Not implemented yet')
            
            hdul.writeto(odir+self.scifiles[i][:-5]+'_klcal.fits', overwrite=True, output_verify='fix')
            hdul.close()
        print('')
        
        data_before = np.array(data_before)
        data_after = np.array(data_after)
        plt.plot(np.mean(data_before, axis=0), label='before')
        plt.plot(np.mean(data_after, axis=0), label='after')
        plt.xlabel('Index')
        plt.ylabel('Kernel phase')
        plt.title('Karhunen-Loeve calibration')
        plt.legend()
        plt.show()
        
        return None
