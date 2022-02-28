from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import corner as cp
import matplotlib.patheffects as PathEffects
import os
import sys

from scipy.interpolate import Rbf

from . import util
from .opticstools import opticstools as ot

pa_mtoc = '-' # model to chip conversion for position angle

datacol = 'mediumaquamarine'
modelcol = 'teal'
gridcol = 'slategray'
modres = 1000

plt.rc('font', size=12)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=18)


# =============================================================================
# MAIN
# =============================================================================

def vis2_ud_base(data_list,
                 fit,
                 smear=None,
                 ofile=None):
    """
    Parameters
    ----------
    data_list: list of dict
        List of data whose squared visibility amplitudes shall be plotted. The
        list contains one data structure for each observation.
    fit: dict
        Uniform disk fit whose squared visibility amplitudes shall be plotted.
    smear: int
        Numerical bandwidth smearing which shall be used.
    ofile: str
        Path under which figures shall be saved.
    """
    
    bb = []
    vis2 = []
    dvis2 = []
    vis2_mod = []
    for i in range(len(data_list)):
        bb += [np.sqrt(data_list[i]['uu'].flatten()**2+data_list[i]['vv'].flatten()**2)]
        vis2 += [data_list[i]['vis2'].flatten()]
        dvis2 += [data_list[i]['dvis2'].flatten()]
        vis_mod = util.vis_ud(p0=fit['p'],
                              data=data_list[i],
                              smear=smear)
        vis2_mod += [util.vis2vis2(vis_mod,
                                   data=data_list[i]).flatten()]
    bb = np.concatenate(bb)
    vis2 = np.concatenate(vis2)
    dvis2 = np.concatenate(dvis2)
    vis2_mod = np.concatenate(vis2_mod)
    vis2_res = vis2-vis2_mod
    
    xmin, xmax = np.min(bb), np.max(bb)
    data = {}
    data['uu'] = np.linspace(xmin, xmax, modres)
    data['vv'] = np.zeros(modres)
    vis_mod = util.vis_ud(p0=fit['p'],
                          data=data,
                          smear=None)
    vis_mod_l = util.vis_ud(p0=fit['p']-fit['dp'],
                            data=data,
                            smear=None)
    vis_mod_u = util.vis_ud(p0=fit['p']+fit['dp'],
                            data=data,
                            smear=None)
    vis2_mod = np.abs(vis_mod)**2
    vis2_mod_l = np.abs(vis_mod_l)**2
    vis2_mod_u = np.abs(vis_mod_u)**2
    
    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(bb/1e6, vis2, yerr=dvis2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0].plot(data['uu']/1e6, vis2_mod, color=modelcol, zorder=4, label='Model')
    ax[0].fill_between(data['uu']/1e6, vis2_mod_l, vis2_mod_u, facecolor=modelcol, alpha=2./3., edgecolor='none', zorder=3)
    temp = ax[0].get_ylim()
    ax[0].axhline(1., ls='--', color=gridcol, zorder=2)
    text = ax[0].text(0.01, 0.01, '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][0], fit['dp'][0]), ha='left', va='bottom', transform=ax[0].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylim(temp)
    ax[0].set_ylabel('$|V|^2$')
    ax[0].legend(loc='upper right')
    ax[1].plot(bb/1e6, vis2_res/dvis2/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Baseline [M$\lambda$]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0., hspace=0.)
    fig.align_ylabels()
    plt.suptitle('Uniform disk fit')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_vis2_ud.pdf')
    # plt.show()
    plt.close()

def vis2_ud(data_list,
            fit,
            smear=None,
            ofile=None):
    """
    Parameters
    ----------
    data_list: list of dict
        List of data whose squared visibility amplitudes shall be plotted. The
        list contains one data structure for each observation.
    fit: dict
        Uniform disk fit whose squared visibility amplitudes shall be plotted.
    smear: int
        Numerical bandwidth smearing which shall be used.
    ofile: str
        Path under which figures shall be saved.
    """
    
    vis2 = []
    dvis2 = []
    vis2_mod = []
    for i in range(len(data_list)):
        vis2 += [data_list[i]['vis2'].flatten()]
        dvis2 += [data_list[i]['dvis2'].flatten()]
        vis_mod = util.vis_ud(p0=fit['p'],
                              data=data_list[i],
                              smear=smear)
        vis2_mod += [util.vis2vis2(vis_mod,
                                   data=data_list[i]).flatten()]
    vis2 = np.concatenate(vis2)
    dvis2 = np.concatenate(dvis2)
    vis2_mod = np.concatenate(vis2_mod)
    vis2_res = vis2-vis2_mod
    
    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(vis2_mod, vis2, yerr=dvis2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0].plot([np.min(vis2_mod), np.max(vis2_mod)], [np.min(vis2_mod), np.max(vis2_mod)], color=modelcol, zorder=4, label='Model')
    temp = ax[0].get_ylim()
    ax[0].axhline(1., ls='--', color=gridcol, zorder=2)
    text = ax[0].text(0.01, 0.01, '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][0], fit['dp'][0]), ha='left', va='bottom', transform=ax[0].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylim(temp)
    ax[0].set_ylabel('Data $|V|^2$')
    ax[0].legend(loc='upper right')
    ax[1].plot(vis2_mod, vis2_res/dvis2/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model $|V|^2$')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0., hspace=0.)
    fig.align_ylabels()
    plt.suptitle('Uniform disk fit')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_vis2_ud.pdf')
    # plt.show()
    plt.close()

def t3_bin(data_list,
           fit,
           smear=None,
           ofile=None):
    """
    Parameters
    ----------
    data_list: list of dict
        List of data whose closure phases shall be plotted. The list contains 
        one data structure for each observation.
    fit: dict
        Point-source companion fit whose closure phase shall be plotted.
    smear: int
        Numerical bandwidth smearing which shall be used.
    ofile: str
        Path under which figures shall be saved.
    """
    
    t3 = []
    dt3 = []
    t3_mod = []
    for i in range(len(data_list)):
        dra = fit['p'][1].copy()
        ddec = fit['p'][2].copy()
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
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp])
        t3 += [data_list[i]['t3'].flatten()]
        dt3 += [data_list[i]['dt3'].flatten()]
        vis_mod = util.vis_bin(p0=p0_temp,
                               data=data_list[i],
                               smear=smear)
        t3_mod += [util.vis2t3(vis_mod,
                               data=data_list[i]).flatten()]
    t3 = np.concatenate(t3)
    dt3 = np.concatenate(dt3)
    t3_mod = np.concatenate(t3_mod)
    t3_res = t3-t3_mod
    
    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(t3_mod, t3, yerr=dt3, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0].plot([np.min(t3_mod), np.max(t3_mod)], [np.min(t3_mod), np.max(t3_mod)], color=modelcol, zorder=4, label='Model')
    ax[0].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[0].text(0.01, 0.01, '$f$ = %.3f +/- %.3f %%' % (fit['p'][0]*100., fit['dp'][0]*100.), ha='left', va='bottom', transform=ax[0].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylabel('Data closure phase [rad]')
    ax[0].legend(loc='upper right')
    ax[1].plot(t3_mod, t3_res/dt3/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model closure phase [rad]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.25, hspace=0.)
    fig.align_ylabels()
    plt.suptitle('Point-source companion fit')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_t3_bin.pdf')
    # plt.show()
    plt.close()

def vis2_t3_ud_bin(data_list,
                   fit,
                   smear=None,
                   ofile=None):
    """
    Parameters
    ----------
    data_list: list of dict
        List of data whose squared visibility amplitudes and closure phases
        shall be plotted. The list contains one data structure for each
        observation.
    fit: dict
        Uniform disk with point-source companion fit whose squared visibility
        amplitudes and closure phases shall be plotted.
    smear: int
        Numerical bandwidth smearing which shall be used.
    ofile: str
        Path under which figures shall be saved.
    """
    
    vis2 = []
    dvis2 = []
    vis2_mod = []
    t3 = []
    dt3 = []
    t3_mod = []
    for i in range(len(data_list)):
        dra = fit['p'][1].copy()
        ddec = fit['p'][2].copy()
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
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp, fit['p'][3].copy()])
        vis2 += [data_list[i]['vis2'].flatten()]
        dvis2 += [data_list[i]['dvis2'].flatten()]
        t3 += [data_list[i]['t3'].flatten()]
        dt3 += [data_list[i]['dt3'].flatten()]
        vis_mod = util.vis_ud_bin(p0=p0_temp,
                                  data=data_list[i],
                                  smear=smear)
        vis2_mod += [util.vis2vis2(vis_mod,
                                   data=data_list[i]).flatten()]
        t3_mod += [util.vis2t3(vis_mod,
                               data=data_list[i]).flatten()]
    vis2 = np.concatenate(vis2)
    dvis2 = np.concatenate(dvis2)
    vis2_mod = np.concatenate(vis2_mod)
    vis2_res = vis2-vis2_mod
    t3 = np.concatenate(t3)
    dt3 = np.concatenate(dt3)
    t3_mod = np.concatenate(t3_mod)
    t3_res = t3-t3_mod
    
    fig, ax = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(9.6, 4.8))
    ax[0, 0].errorbar(vis2_mod, vis2, yerr=dvis2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0, 0].plot([np.min(vis2_mod), np.max(vis2_mod)], [np.min(vis2_mod), np.max(vis2_mod)], color=modelcol, zorder=4, label='Model')
    temp = ax[0, 0].get_ylim()
    ax[0, 0].axhline(1., ls='--', color=gridcol, zorder=2)
    text = ax[0, 0].text(0.01, 0.01, '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][3], fit['dp'][3]), ha='left', va='bottom', transform=ax[0, 0].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0, 0].set_ylim(temp)
    ax[0, 0].set_ylabel('Data $|V|^2$')
    ax[0, 0].legend(loc='upper right')
    ax[1, 0].plot(vis2_mod, vis2_res/dvis2/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1, 0].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1, 0].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1, 0].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1, 0].get_ylim()))
    ax[1, 0].set_ylim(-ylim, ylim)
    ax[1, 0].set_xlabel('Model $|V|^2$')
    ax[1, 0].set_ylabel('Res. [$\sigma$/$\chi$]')
    ax[0, 1].errorbar(t3_mod, t3, yerr=dt3, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0, 1].plot([np.min(t3_mod), np.max(t3_mod)], [np.min(t3_mod), np.max(t3_mod)], color=modelcol, zorder=4, label='Model')
    ax[0, 1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[0, 1].text(0.01, 0.01, '$f$ = %.3f +/- %.3f %%' % (fit['p'][0]*100., fit['dp'][0]*100.), ha='left', va='bottom', transform=ax[0, 1].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0, 1].set_ylabel('Data closure phase [rad]')
    ax[0, 1].legend(loc='upper right')
    ax[1, 1].plot(t3_mod, t3_res/dt3/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1, 1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1, 1].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1, 1].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1, 1].get_ylim()))
    ax[1, 1].set_ylim(-ylim, ylim)
    ax[1, 1].set_xlabel('Model closure phase [rad]')
    ax[1, 1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=1./3., hspace=0.)
    fig.align_ylabels()
    plt.suptitle('Uniform disk with point-source companion fit')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_vis2_t3_ud_bin.pdf')
    # plt.show()
    plt.close()

def kp_bin(data_list,
           fit,
           smear=None,
           ofile=None):
    """
    Parameters
    ----------
    data_list: list of dict
        List of data whose kernel phases shall be plotted. The list contains 
        one data structure for each observation.
    fit: dict
        Point-source companion fit whose kernel phase shall be plotted.
    smear: int
        Numerical bandwidth smearing which shall be used.
    ofile: str
        Path under which figures shall be saved.
    """
    
    kp = []
    dkp = []
    kp_mod = []
    for i in range(len(data_list)):
        dra = fit['p'][1].copy()
        ddec = fit['p'][2].copy()
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
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp])
        kp += [data_list[i]['kp'].flatten()]
        dkp += [data_list[i]['dkp'].flatten()]
        vis_mod = util.vis_bin(p0=p0_temp,
                               data=data_list[i],
                               smear=smear)
        kp_mod += [util.vis2kp(vis_mod,
                               data=data_list[i]).flatten()]
    kp = np.concatenate(kp)
    dkp = np.concatenate(dkp)
    kp_mod = np.concatenate(kp_mod)
    kp_res = kp-kp_mod
    
    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(kp_mod, kp, yerr=dkp, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data')
    ax[0].plot([np.min(kp_mod), np.max(kp_mod)], [np.min(kp_mod), np.max(kp_mod)], color=modelcol, zorder=4, label='Model')
    ax[0].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[0].text(0.01, 0.01, '$f$ = %.3f +/- %.3f %%' % (fit['p'][0]*100., fit['dp'][0]*100.), ha='left', va='bottom', transform=ax[0].transAxes, zorder=5)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylabel('Data kernel phase [rad]')
    ax[0].legend(loc='upper right')
    ax[1].plot(kp_mod, kp_res/dkp/np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0., ls='--', color=gridcol, zorder=2)
    text = ax[1].text(0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model kernel phase [rad]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.25, hspace=0.)
    fig.align_ylabels()
    plt.suptitle('Point-source companion fit')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_kp_bin.pdf')
    # plt.show()
    plt.close()

def lincmap(pps,
            pes,
            chi2s,
            fit,
            sep_range,
            step_size,
            ofile=None):
    """
    Parameters
    ----------
    pps: array
        Array of shape (model parameters x RA steps x DEC steps) containing
        best fit model parameters for grid.
    pes: array
        Array of shape (model parameters x RA steps x DEC steps) containing
        uncertainties of best fit model parameters for grid.
    chi2s: array
        Array of shape (1 x RA steps x DEC steps) containing best fit
        chi-squared for the grid.
    fit: dict
        Model fit whose chi-squared map shall be plotted.
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    ofile: str
        Path under which figures shall be saved.
    """
    
    grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range,
                                             step_size=step_size,
                                             verbose=False)
    emax = np.nanmax(grid_ra_dec[0])
    sep = np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
    pa = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
    rad, max = ot.azimuthalAverage(np.abs(pps[0]), returnradii=True, binsize=1., return_max=True)
    avg = ot.azimuthalAverage(np.abs(pps[0]), binsize=1.)
    rad *= step_size
    
    fig, ax = plt.subplots(1, 3, figsize=(19.2, 4.8))
    temp = pps[0]
    temp[temp <= 0.] = np.min(temp[temp > 0.])
    p0 = ax[0].imshow(np.log10(temp), cmap='hot', vmin=-4, vmax=-1, origin='lower', extent=(emax+step_size/2., -emax-step_size/2., -emax-step_size/2., emax+step_size/2.))
    c0 = plt.colorbar(p0, ax=ax[0])
    c0.set_label(r'$\mathrm{log_{10}}$(relative flux)', rotation=270, labelpad=20)
    ax[0].plot(0., 0., marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='white', lw=5, fill=False)
    ax[0].add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='black', lw=2.5, fill=False)
    ax[0].add_artist(cc)
    text = ax[0].text(0.01, 0.99, '$f$ = %.3e +/- %.3e %%' % (fit['p'][0], fit['dp'][0]), ha='left', va='top', color='black', transform=ax[0].transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[0].text(0.01, 0.06, '$\\rho$ = %.1f mas' % sep, ha='left', va='bottom', color='black', transform=ax[0].transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[0].text(0.01, 0.01, '$\\varphi$ = %.1f deg' % pa, ha='left', va='bottom', color='black', transform=ax[0].transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_xlabel('$\Delta$RA [mas]')
    ax[0].set_ylabel('$\Delta$DEC [mas]')
    ax[0].set_title('Linear contrast map')
    p1 = ax[1].imshow(chi2s/fit['ndof'], cmap='cubehelix', origin='lower', extent=(emax+step_size/2., -emax-step_size/2., -emax-step_size/2., emax+step_size/2.))
    c1 = plt.colorbar(p1, ax=ax[1])
    c1.set_label('$\chi^2$', rotation=270, labelpad=20)
    ax[1].plot(0., 0., marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='white', lw=5, fill=False)
    ax[1].add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='black', lw=2.5, fill=False)
    ax[1].add_artist(cc)
    text = ax[1].text(0.99, 0.99, '$N_{\sigma}$ = %.1f' % fit['nsigma'], ha='right', va='top', color='black', transform=ax[1].transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[1].text(0.99, 0.01, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='bottom', color='black', transform=ax[1].transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[1].set_xlabel('$\Delta$RA [mas]')
    ax[1].set_ylabel('$\Delta$DEC [mas]')
    ax[1].set_title('Chi-squared map')
    ax[2].plot(rad, max, label='max')
    ax[2].plot(rad, avg, label='avg')
    ax[2].grid(axis='y')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('Separation [mas]')
    ax[2].set_ylabel('Contrast')
    ax[2].set_title('Detection limits')
    ax[2].legend(loc='upper right')
    # ax = ax[2].twinx()
    # ax.plot(rad, max/avg, color='black', ls=':')
    # ax.set_ylabel(r'Significance [$\sigma$]', rotation=270, labelpad=20)
    plt.tight_layout()
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_lincmap.pdf')
    # plt.show()
    plt.close()

def chi2map(pps_unique,
            chi2s_unique,
            fit,
            sep_range,
            step_size,
            ofile=None):
    """
    Parameters
    ----------
    pps_unique: array
        Array of shape (unique minima x model parameters) containing best fit
        model parameters for each unique minimum.
    chi2s_unique: array
        Array of shape (unique minima) containing best fit chi-squared for
        each unique minimum.
    fit: dict
        Model fit whose chi-squared map shall be plotted.
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    ofile: str
        Path under which figures shall be saved.
    """
    
    grid_ra_dec_fine, grid_sep_pa_fine = util.get_grid(sep_range=sep_range,
                                                       step_size=step_size/4.,
                                                       verbose=False)
    emax = np.nanmax(grid_ra_dec_fine[0])
    func = Rbf(pps_unique[:, 1], pps_unique[:, 2], chi2s_unique, function='linear')
    chi2s_rbf = func(grid_ra_dec_fine[0].flatten(), grid_ra_dec_fine[1].flatten()).reshape(grid_ra_dec_fine[0].shape)
    sep = np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
    pa = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
    dsep = np.sqrt((fit['p'][1]/sep*fit['dp'][1])**2+(fit['p'][2]/sep*fit['dp'][2])**2)
    dpa = np.rad2deg(np.sqrt((fit['p'][2]/sep**2*fit['dp'][1])**2+(-fit['p'][1]/sep**2*fit['dp'][2])**2))
    
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    p0 = ax.imshow(chi2s_rbf/fit['ndof'], cmap='cubehelix', origin='lower', extent=(emax+step_size/2., -emax-step_size/2., -emax-step_size/2., emax+step_size/2.))
    c0 = plt.colorbar(p0, ax=ax)
    c0.set_label('$\chi^2$', rotation=270, labelpad=20)
    ax.plot(0., 0., marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='white', lw=5, fill=False)
    ax.add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax/10., color='black', lw=2.5, fill=False)
    ax.add_artist(cc)
    text = ax.text(0.01, 0.99, '$f$ = %.3f +/- %.3f %%' % (fit['p'][0]*100., fit['dp'][0]*100.), ha='left', va='top', color='black', transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(0.01, 0.06, '$\\rho$ = %.1f +/- %.1f mas' % (sep, dsep), ha='left', va='bottom', color='black', transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(0.01, 0.01, '$\\varphi$ = %.1f +/- %.1f deg' % (pa, dpa), ha='left', va='bottom', color='black', transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(0.99, 0.99, '$N_{\sigma}$ = %.1f' % fit['nsigma'], ha='right', va='top', color='black', transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(0.99, 0.01, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='bottom', color='black', transform=ax.transAxes)
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$\Delta$RA [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    plt.suptitle('Chi-squared map')
    if (ofile is not None):
        index = ofile.rfind('/')
        if index != -1:
            temp = ofile[:index]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
        plt.savefig(ofile+'_chi2map.pdf')
    # plt.show()
    plt.close()

def chains(fit,
           sampler,
           ofile=None):
    """
    Parameters
    ----------
    fit: dict
        Model fit whose MCMC chains shall be plotted.
    sampler: array
        Sampler of MCMC.
    ofile: str
        Path under which figures shall be saved.
    """
    
    if (fit['model'] == 'ud'):
        fig = plt.figure()
        plt.plot(sampler.flatchain[:, 0], color=datacol)
        plt.axhline(np.percentile(sampler.flatchain[:, 0], 50.), color=modelcol, label='MCMC median')
        plt.axhline(fit['p'][0], ls='--', color=gridcol, label='Initial guess')
        plt.xlabel('Step')
        plt.ylabel('$\\theta$ [mas]')
        plt.legend(loc='upper right')
        plt.suptitle('MCMC chains')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_chains.pdf')
        # plt.show()
        plt.close()
    elif (fit['model'] == 'bin'):
        ylabels = ['$f$ [%]', '$\\rho$ [mas]', '$\\varphi$ [deg]']
        rho = np.sqrt(sampler.flatchain[:, 1]**2+sampler.flatchain[:, 2]**2)
        rho0 = np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
        phi = np.rad2deg(np.arctan2(sampler.flatchain[:, 1], sampler.flatchain[:, 2]))
        phi0 = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
        fig, ax = plt.subplots(len(fit['p']), 1, sharex='col', figsize=(6.4, 1.6*len(fit['p'])))
        ax[0].plot(sampler.flatchain[:, 0]*100., color=datacol)
        ax[0].axhline(np.percentile(sampler.flatchain[:, 0], 50.)*100., color=modelcol, label='MCMC median')
        ax[0].axhline(fit['p'][0]*100., ls='--', color=gridcol, label='Initial guess')
        ax[1].plot(rho, color=datacol)
        ax[1].axhline(np.percentile(rho, 50.), color=modelcol)
        ax[1].axhline(rho0, ls='--', color=gridcol)
        ax[2].plot(phi, color=datacol)
        ax[2].axhline(np.percentile(phi, 50.), color=modelcol)
        ax[2].axhline(phi0, ls='--', color=gridcol)
        plt.xlabel('Step')
        for i in range(len(fit['p'])):
            ax[i].set_ylabel(ylabels[i])
            if (i == 0):
                ax[i].legend(loc='upper right')
        plt.subplots_adjust(wspace=0.25, hspace=0.)
        fig.align_ylabels()
        plt.suptitle('MCMC chains')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_chains.pdf')
        # plt.show()
        plt.close()
    else:
        ylabels = ['$f$ [%]', '$\\rho$ [mas]', '$\\varphi$ [deg]', '$\\theta$ [mas]']
        rho = np.sqrt(sampler.flatchain[:, 1]**2+sampler.flatchain[:, 2]**2)
        rho0 = np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
        phi = np.rad2deg(np.arctan2(sampler.flatchain[:, 1], sampler.flatchain[:, 2]))
        phi0 = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
        fig, ax = plt.subplots(len(fit['p']), 1, sharex='col', figsize=(6.4, 1.6*len(fit['p'])))
        ax[0].plot(sampler.flatchain[:, 0]*100., color=datacol)
        ax[0].axhline(np.percentile(sampler.flatchain[:, 0], 50.)*100., color=modelcol, label='MCMC median')
        ax[0].axhline(fit['p'][0]*100., ls='--', color=gridcol, label='Initial guess')
        ax[1].plot(rho, color=datacol)
        ax[1].axhline(np.percentile(rho, 50.), color=modelcol)
        ax[1].axhline(rho0, ls='--', color=gridcol)
        ax[2].plot(phi, color=datacol)
        ax[2].axhline(np.percentile(phi, 50.), color=modelcol)
        ax[2].axhline(phi0, ls='--', color=gridcol)
        ax[3].plot(sampler.flatchain[:, 3], color=datacol)
        ax[3].axhline(np.percentile(sampler.flatchain[:, 3], 50.), color=modelcol)
        ax[3].axhline(fit['p'][3], ls='--', color=gridcol)
        plt.xlabel('Step')
        for i in range(len(fit['p'])):
            ax[i].set_ylabel(ylabels[i])
            if (i == 0):
                ax[i].legend(loc='upper right')
        plt.subplots_adjust(wspace=0.25, hspace=0.)
        fig.align_ylabels()
        plt.suptitle('MCMC chains')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_chains.pdf')
        # plt.show()
        plt.close()

def corner(fit,
           sampler,
           ofile=None):
    """
    Parameters
    ----------
    fit: dict
        Model fit whose posterior distribution shall be plotted.
    sampler: array
        Sampler of MCMC.
    ofile: str
        Path under which figures shall be saved.
    """
    
    if (fit['model'] == 'ud'):
        fig = cp.corner(sampler.flatchain,
                        labels=[r'$\theta$ [mas]'],
                        titles=[r'$\theta$'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt='.3f')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_corner.pdf')
        # plt.show()
        plt.close()
    elif (fit['model'] == 'bin'):
        temp = sampler.flatchain.copy()
        temp[:, 0] *= 100.
        temp[:, 1] = np.sqrt(sampler.flatchain[:, 1]**2+sampler.flatchain[:, 2]**2)
        temp[:, 2] = np.rad2deg(np.arctan2(sampler.flatchain[:, 1], sampler.flatchain[:, 2]))
        fig = cp.corner(temp,
                        labels=[r'$f$ [%]', r'$\rho$ [mas]', r'$\varphi$ [deg]'],
                        titles=[r'$f$', r'$\rho$', r'$\varphi$'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt='.3f')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_corner.pdf')
        # plt.show()
        plt.close()
    else:
        temp = sampler.flatchain.copy()
        temp[:, 0] *= 100.
        temp[:, 1] = np.sqrt(sampler.flatchain[:, 1]**2+sampler.flatchain[:, 2]**2)
        temp[:, 2] = np.rad2deg(np.arctan2(sampler.flatchain[:, 1], sampler.flatchain[:, 2]))
        fig = cp.corner(temp,
                        labels=[r'$f$ [%]', r'$\rho$ [mas]', r'$\varphi$ [deg]', r'$\theta$ [mas]'],
                        titles=[r'$f$', r'$\rho$', r'$\varphi$', r'$\theta$'],
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt='.3f')
        if (ofile is not None):
            index = ofile.rfind('/')
            if index != -1:
                temp = ofile[:index]
                if (not os.path.exists(temp)):
                    os.makedirs(temp)
            plt.savefig(ofile+'_mcmc_corner.pdf')
        # plt.show()
        plt.close()
