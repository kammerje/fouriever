from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import numpy as np

import glob
import os

from . import inst

observables_known = ['v2', 'cp', 'kp']


# =============================================================================
# MAIN
# =============================================================================


class data:
    def __init__(self, idir, fitsfiles):
        """
        Parameters
        ----------
        idir: str
            Input directory where fits files are located.
        fitsfiles: list of str, None
            List of fits files which shall be opened. All fits files from
            ``idir`` are opened with ``fitsfiles=None``.
        """

        self.idir = idir
        self.fitsfiles = fitsfiles

        if self.fitsfiles is None:
            self.fitsfiles = glob.glob(self.idir + '*fits')
            for i, item in enumerate(self.fitsfiles):
                head, tail = os.path.split(item)
                self.fitsfiles[i] = tail

        self.inst_list = []
        self.data_list = []
        for i in range(len(self.fitsfiles)):
            inst_list, data_list = inst.open(idir=idir, fitsfile=self.fitsfiles[i], verbose=False)
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

    def set_inst(self, inst):
        """
        Parameters
        ----------
        inst: str
            Instrument which shall be selected.
        """

        if inst in self.inst_list:
            self.inst = inst
            print('Selected instrument = ' + self.inst)
            print('   Use self.set_inst(inst) to change the selected instrument')
        else:
            raise UserWarning(inst + ' is an unknown instrument')

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
            while j < len(ww) and flag:
                if observables_known[i] not in self.data_list[ww[j]][0].keys():
                    flag = False
                j += 1
            if flag:
                observables += [observables_known[i]]

        return observables

    def set_observables(self, observables):
        """
        Parameters
        ----------
        observables: list of str
            List of observables which shall be selected.
        """

        observables_valid = self.get_observables()
        for i in range(len(observables)):
            if observables[i] not in observables_valid:
                raise UserWarning(observables[i] + ' is not a valid observable for the currently selected instrument')
        self.observables = observables
        print('Selected observables = ' + str(self.observables))
        print('   Use self.set_observables(observables) to change the selected observables')

        return None

    def clear_cov(self):
        """ """

        for i in range(len(self.fitsfiles)):
            hdul = pyfits.open(os.path.join(self.idir, self.fitsfiles[i]))
            try:
                hdul.pop('V2COV')
            except KeyError:
                pass
            try:
                hdul.pop('CPCOV')
            except KeyError:
                pass
            hdul.writeto(os.path.join(self.idir, self.fitsfiles[i]), output_verify='fix', overwrite=True)

        pass

    def add_v2cov(self, odir):
        """
        Parameters
        ----------
        odir: str
            Output directory where fits files with covariance shall be saved
            to.
        """

        print('   Computing visibility amplitude correlations')

        if not os.path.exists(odir):
            os.makedirs(odir)

        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            data_list += [self.data_list[ww[i]]]

        if len(self.fitsfiles) != len(data_list):
            raise UserWarning('All input fits files should contain data of the selected instrument')

        for i in range(len(self.fitsfiles)):
            Nwave = data_list[i][0]['wave'].shape[0]
            Nbase = data_list[i][0]['v2'].shape[0]

            cor = np.diag(np.ones(Nwave * Nbase))
            covs = []
            for j in range(len(data_list[i])):
                dv2 = data_list[i][j]['dv2']
                cov = np.multiply(cor, dv2.flatten()[:, None] * dv2.flatten()[None, :])
                covs += [cov]
            covs = np.array(covs)

            hdul = pyfits.open(os.path.join(self.idir, self.fitsfiles[i]))
            hdu0 = pyfits.ImageHDU(covs)
            hdu0.header['EXTNAME'] = 'V2COV'
            hdu0.header['INSNAME'] = self.inst
            hdul += [hdu0]
            hdul.writeto(odir + self.fitsfiles[i], output_verify='fix', overwrite=True)

        # plt.imshow(cor, origin='lower')
        # plt.xlabel('Index')
        # plt.ylabel('Index')
        # plt.title('Visibility amplitude correlation')
        # plt.show()
        # plt.close()

        return None

    def add_cpcov(self, odir):
        """
        Parameters
        ----------
        odir: str
            Output directory where fits files with covariance shall be saved
            to.
        """

        print('   Computing closure phase correlations')

        if not os.path.exists(odir):
            os.makedirs(odir)

        data_list = []
        ww = np.where(np.array(self.inst_list) == self.inst)[0]
        for i in range(len(ww)):
            data_list += [self.data_list[ww[i]]]

        if len(self.fitsfiles) != len(data_list):
            raise UserWarning('All input fits files should contain data of the selected instrument')

        for i in range(len(self.fitsfiles)):
            cpmat = data_list[i][0]['cpmat'].copy()
            Nwave = data_list[i][0]['wave'].shape[0]
            Nbase = cpmat.shape[1]
            Ntria = cpmat.shape[0]

            trafo = np.zeros((Nwave * Ntria, Nwave * Nbase))
            for k in range(Ntria):
                for l in range(Nbase):
                    trafo[k * Nwave : (k + 1) * Nwave, l * Nwave : (l + 1) * Nwave] = (
                        np.diag(np.ones(Nwave)) * cpmat[k, l]
                    )

            cor = np.dot(trafo, np.dot(np.diag(np.ones(Nwave * Nbase)), trafo.T)) / 3.0
            covs = []
            for j in range(len(data_list[i])):
                dcp = data_list[i][j]['dcp']
                cov = np.multiply(cor, dcp.flatten()[:, None] * dcp.flatten()[None, :])
                covs += [cov]
            covs = np.array(covs)

            hdul = pyfits.open(os.path.join(self.idir, self.fitsfiles[i]))
            hdu0 = pyfits.ImageHDU(covs)
            hdu0.header['EXTNAME'] = 'CPCOV'
            hdu0.header['INSNAME'] = self.inst
            hdul += [hdu0]
            hdul.writeto(odir + self.fitsfiles[i], output_verify='fix', overwrite=True)

        # plt.imshow(cor, origin='lower')
        # plt.xlabel('Index')
        # plt.ylabel('Index')
        # plt.title('Closure phase correlation')
        # plt.show()
        # plt.close()

        return None

    def add_cov(self, odir):
        """
        Parameters
        ----------
        odir: str
            Output directory where fits files with covariance shall be saved
            to.
        """

        print('Computing correlations')

        if not os.path.exists(odir):
            os.makedirs(odir)

        self.add_v2cov(odir=odir)

        temp = self.idir
        self.idir = odir
        self.add_cpcov(odir=odir)
        self.idir = temp

        return None
