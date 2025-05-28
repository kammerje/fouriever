from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import numpy as np

rad2mas = 180.0 / np.pi * 3600.0 * 1000.0  # convert rad to mas
mas2rad = np.pi / 180.0 / 3600.0 / 1000.0  # convert mas to rad
pa_mtoc = '-'  # model to chip conversion for position angle


# =============================================================================
# MAIN
# =============================================================================


def average_single(fitsfile):
    fitsfile = str(fitsfile)  # In case a Path is passed

    out_path = fitsfile.replace('.oifits', '_avg.oifits')
    hdul = pyfits.open(fitsfile)

    oi_wavelength = pyfits.BinTableHDU.from_columns(
        pyfits.ColDefs(
            (
                pyfits.Column(
                    name='EFF_WAVE',
                    format='1E',
                    unit='METERS',
                    array=np.nanmean(hdul['OI_WAVELENGTH'].data['EFF_WAVE'], keepdims=True),
                ),
                pyfits.Column(
                    name='EFF_BAND',
                    format='1E',
                    unit='METERS',
                    array=np.nanmean(hdul['OI_WAVELENGTH'].data['EFF_BAND'], keepdims=True),
                ),
            )
        )
    )
    oi_wavelength.header['EXTNAME'] = 'OI_WAVELENGTH'
    oi_wavelength.header['INSNAME'] = hdul['OI_WAVELENGTH'].header['INSNAME']

    oi_vis2 = pyfits.BinTableHDU.from_columns(
        pyfits.ColDefs(
            [
                pyfits.Column(name='TARGET_ID', format='1I', array=np.array(hdul['OI_VIS2'].data['TARGET_ID'])),
                pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=np.array(hdul['OI_VIS2'].data['TIME'])),
                pyfits.Column(name='MJD', unit='DAY', format='1D', array=np.array(hdul['OI_VIS2'].data['MJD'])),
                pyfits.Column(
                    name='INT_TIME', format='1D', unit='SECONDS', array=np.array(hdul['OI_VIS2'].data['INT_TIME'])
                ),
                pyfits.Column(name='VIS2DATA', format='1D', array=np.nanmean(hdul['OI_VIS2'].data['VIS2DATA'], axis=1)),
                pyfits.Column(
                    name='VIS2ERR',
                    format='1D',
                    array=np.nanmean(hdul['OI_VIS2'].data['VIS2ERR'], axis=1)
                    / np.sqrt(hdul['OI_VIS2'].data['VIS2ERR'].shape[1]),
                ),
                pyfits.Column(
                    name='UCOORD', format='1D', unit='METERS', array=np.array(hdul['OI_VIS2'].data['UCOORD'])
                ),
                pyfits.Column(
                    name='VCOORD', format='1D', unit='METERS', array=np.array(hdul['OI_VIS2'].data['VCOORD'])
                ),
                pyfits.Column(name='STA_INDEX', format='2I', array=np.array(hdul['OI_VIS2'].data['STA_INDEX'])),
                pyfits.Column(name='FLAG', format='1L', array=np.min(hdul['OI_VIS2'].data['FLAG'], axis=1)),
            ]
        )
    )
    oi_vis2.header['EXTNAME'] = 'OI_VIS2'
    oi_vis2.header['INSNAME'] = hdul['OI_VIS2'].header['INSNAME']

    oi_t3 = pyfits.BinTableHDU.from_columns(
        pyfits.ColDefs(
            (
                pyfits.Column(name='TARGET_ID', format='1I', array=np.array(hdul['OI_T3'].data['TARGET_ID'])),
                pyfits.Column(name='TIME', format='1D', unit='SECONDS', array=np.array(hdul['OI_T3'].data['TIME'])),
                pyfits.Column(name='MJD', format='1D', unit='DAY', array=np.array(hdul['OI_T3'].data['MJD'])),
                pyfits.Column(
                    name='INT_TIME', format='1D', unit='SECONDS', array=np.array(hdul['OI_T3'].data['INT_TIME'])
                ),
                pyfits.Column(name='T3AMP', format='1D', array=np.nanmean(hdul['OI_T3'].data['T3AMP'], axis=1)),
                pyfits.Column(
                    name='T3AMPERR',
                    format='1D',
                    array=np.nanmean(hdul['OI_T3'].data['T3AMPERR'], axis=1)
                    / np.sqrt(hdul['OI_T3'].data['T3AMPERR'].shape[1]),
                ),
                pyfits.Column(
                    name='T3PHI', format='1D', unit='DEGREES', array=np.nanmean(hdul['OI_T3'].data['T3PHI'], axis=1)
                ),
                pyfits.Column(
                    name='T3PHIERR',
                    format='1D',
                    unit='DEGREES',
                    array=np.nanmean(hdul['OI_T3'].data['T3PHIERR'], axis=1)
                    / np.sqrt(hdul['OI_T3'].data['T3PHIERR'].shape[1]),
                ),
                pyfits.Column(
                    name='U1COORD', format='1D', unit='METERS', array=np.array(hdul['OI_T3'].data['U1COORD'])
                ),
                pyfits.Column(
                    name='V1COORD', format='1D', unit='METERS', array=np.array(hdul['OI_T3'].data['V1COORD'])
                ),
                pyfits.Column(
                    name='U2COORD', format='1D', unit='METERS', array=np.array(hdul['OI_T3'].data['U2COORD'])
                ),
                pyfits.Column(
                    name='V2COORD', format='1D', unit='METERS', array=np.array(hdul['OI_T3'].data['V2COORD'])
                ),
                pyfits.Column(name='STA_INDEX', format='3I', array=np.array(hdul['OI_T3'].data['STA_INDEX'])),
                pyfits.Column(name='FLAG', format='1L', array=np.min(hdul['OI_T3'].data['FLAG'], axis=1)),
            )
        )
    )
    oi_t3.header['EXTNAME'] = 'OI_T3'
    oi_t3.header['INSNAME'] = hdul['OI_T3'].header['INSNAME']

    hdul.pop('OI_WAVELENGTH')
    hdul.pop('OI_VIS2')
    hdul.pop('OI_T3')
    hdul.append(oi_wavelength)
    hdul.append(oi_vis2)
    hdul.append(oi_t3)
    hdul.writeto(out_path, output_verify='fix', overwrite=True)
    hdul.close()

    return out_path
