from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import warnings
import matplotlib.pyplot as plt
import numpy as np

import corner as cp
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import os

from scipy.interpolate import Rbf

from . import util
from .opticstools import opticstools as ot

pa_mtoc = '-'  # model to chip conversion for position angle

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
def v2_ud_base(data_list, fit, smear=None, ofile=None):
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
    v2 = []
    dv2 = []
    v2_mod = []
    for i in range(len(data_list)):
        bb += [np.sqrt(data_list[i]['uu'].flatten() ** 2 + data_list[i]['vv'].flatten() ** 2)]
        v2 += [data_list[i]['v2'].flatten()]
        dv2 += [data_list[i]['dv2'].flatten()]
        vis_mod = util.vis_ud(p0=fit['p'], data=data_list[i], smear=smear)
        v2_mod += [util.v2v2(vis_mod, data=data_list[i]).flatten()]
    bb = np.concatenate(bb)
    v2 = np.concatenate(v2)
    dv2 = np.concatenate(dv2)
    v2_mod = np.concatenate(v2_mod)
    v2_res = v2 - v2_mod

    xmin, xmax = np.min(bb), np.max(bb)
    data = {}
    data['uu'] = np.linspace(xmin, xmax, modres)
    data['vv'] = np.zeros(modres)
    vis_mod = util.vis_ud(p0=fit['p'], data=data, smear=None)
    vis_mod_l = util.vis_ud(p0=fit['p'] - fit['dp'], data=data, smear=None)
    vis_mod_u = util.vis_ud(p0=fit['p'] + fit['dp'], data=data, smear=None)
    v2_mod = np.abs(vis_mod) ** 2
    v2_mod_l = np.abs(vis_mod_l) ** 2
    v2_mod_u = np.abs(vis_mod_u) ** 2

    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(
        bb / 1e6, v2, yerr=dv2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0].plot(data['uu'] / 1e6, v2_mod, color=modelcol, zorder=4, label='Model')
    ax[0].fill_between(
        data['uu'] / 1e6, v2_mod_l, v2_mod_u, facecolor=modelcol, alpha=2.0 / 3.0, edgecolor='none', zorder=3
    )
    temp = ax[0].get_ylim()
    ax[0].axhline(1.0, ls='--', color=gridcol, zorder=2)
    text = ax[0].text(
        0.01,
        0.01,
        '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][0], fit['dp'][0]),
        ha='left',
        va='bottom',
        transform=ax[0].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylim(temp)
    ax[0].set_ylabel('$|V|^2$')
    ax[0].legend(loc='upper right')
    ax[1].plot(bb / 1e6, v2_res / dv2 / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Baseline [M$\lambda$]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.align_ylabels()
    plt.suptitle('Uniform disk fit')
    util.save_ofile(ofile, 'v2_ud')
    # plt.show()
    plt.close()


def v2_ud(data_list, fit, smear=None, ofile=None):
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

    v2 = []
    dv2 = []
    v2_mod = []
    for i in range(len(data_list)):
        v2 += [data_list[i]['v2'].flatten()]
        dv2 += [data_list[i]['dv2'].flatten()]
        vis_mod = util.vis_ud(p0=fit['p'], data=data_list[i], smear=smear)
        v2_mod += [util.v2v2(vis_mod, data=data_list[i]).flatten()]
    v2 = np.concatenate(v2)
    dv2 = np.concatenate(dv2)
    v2_mod = np.concatenate(v2_mod)
    v2_res = v2 - v2_mod

    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(
        v2_mod, v2, yerr=dv2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0].plot(
        [np.min(v2_mod), np.max(v2_mod)], [np.min(v2_mod), np.max(v2_mod)], color=modelcol, zorder=4, label='Model'
    )
    temp = ax[0].get_ylim()
    ax[0].axhline(1.0, ls='--', color=gridcol, zorder=2)
    text = ax[0].text(
        0.01,
        0.01,
        '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][0], fit['dp'][0]),
        ha='left',
        va='bottom',
        transform=ax[0].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylim(temp)
    ax[0].set_ylabel('Data $|V|^2$')
    ax[0].legend(loc='upper left')
    ax[1].plot(v2_mod, v2_res / dv2 / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model $|V|^2$')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.align_ylabels()
    plt.suptitle('Uniform disk fit')
    util.save_ofile(ofile, 'v2_ud')
    # plt.show()
    plt.close()


def cp_bin(data_list, fit, smear=None, ofile=None):
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

    cp = []
    dcp = []
    cp_mod = []
    for i in range(len(data_list)):
        dra = fit['p'][1].copy()
        ddec = fit['p'][2].copy()
        rho = np.sqrt(dra**2 + ddec**2)
        phi = np.rad2deg(np.arctan2(dra, ddec))
        if pa_mtoc == '-':
            phi -= data_list[i]['pa']
        elif pa_mtoc == '+':
            phi += data_list[i]['pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        phi = ((phi + 180.0) % 360.0) - 180.0
        dra_temp = rho * np.sin(np.deg2rad(phi))
        ddec_temp = rho * np.cos(np.deg2rad(phi))
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp])
        cp += [data_list[i]['cp'].flatten()]
        dcp += [data_list[i]['dcp'].flatten()]
        vis_mod = util.vis_bin(p0=p0_temp, data=data_list[i], smear=smear)
        cp_mod += [util.v2cp(vis_mod, data=data_list[i]).flatten()]
    cp = np.concatenate(cp)
    dcp = np.concatenate(dcp)
    cp_mod = np.concatenate(cp_mod)
    cp_res = cp - cp_mod

    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(
        cp_mod, cp, yerr=dcp, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0].plot(
        [np.min(cp_mod), np.max(cp_mod)], [np.min(cp_mod), np.max(cp_mod)], color=modelcol, zorder=4, label='Model'
    )
    ax[0].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[0].text(
        0.01,
        0.01,
        '$f$ = %.3f +/- %.3f %%' % (fit['p'][0] * 100.0, fit['dp'][0] * 100.0),
        ha='left',
        va='bottom',
        transform=ax[0].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylabel('Data closure phase [rad]')
    ax[0].legend(loc='upper left')
    ax[1].plot(cp_mod, cp_res / dcp / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model closure phase [rad]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.25, hspace=0.0)
    fig.align_ylabels()
    plt.suptitle('Point-source companion fit')
    util.save_ofile(ofile, 'cp_bin')
    # plt.show()
    plt.close()


def v2_cp_ud_bin(data_list, fit, smear=None, ofile=None):
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

    v2 = []
    dv2 = []
    v2_mod = []
    cp = []
    dcp = []
    cp_mod = []
    for i in range(len(data_list)):
        dra = fit['p'][1].copy()
        ddec = fit['p'][2].copy()
        rho = np.sqrt(dra**2 + ddec**2)
        phi = np.rad2deg(np.arctan2(dra, ddec))
        if pa_mtoc == '-':
            phi -= data_list[i]['pa']
        elif pa_mtoc == '+':
            phi += data_list[i]['pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        phi = ((phi + 180.0) % 360.0) - 180.0
        dra_temp = rho * np.sin(np.deg2rad(phi))
        ddec_temp = rho * np.cos(np.deg2rad(phi))
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp, fit['p'][3].copy()])
        v2 += [data_list[i]['v2'].flatten()]
        dv2 += [data_list[i]['dv2'].flatten()]
        cp += [data_list[i]['cp'].flatten()]
        dcp += [data_list[i]['dcp'].flatten()]
        vis_mod = util.vis_ud_bin(p0=p0_temp, data=data_list[i], smear=smear)
        v2_mod += [util.v2v2(vis_mod, data=data_list[i]).flatten()]
        cp_mod += [util.v2cp(vis_mod, data=data_list[i]).flatten()]
    v2 = np.concatenate(v2)
    dv2 = np.concatenate(dv2)
    v2_mod = np.concatenate(v2_mod)
    v2_res = v2 - v2_mod
    cp = np.concatenate(cp)
    dcp = np.concatenate(dcp)
    cp_mod = np.concatenate(cp_mod)
    cp_res = cp - cp_mod

    fig, ax = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(9.6, 4.8))
    ax[0, 0].errorbar(
        v2_mod, v2, yerr=dv2, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0, 0].plot(
        [np.min(v2_mod), np.max(v2_mod)], [np.min(v2_mod), np.max(v2_mod)], color=modelcol, zorder=4, label='Model'
    )
    temp = ax[0, 0].get_ylim()
    ax[0, 0].axhline(1.0, ls='--', color=gridcol, zorder=2)
    text = ax[0, 0].text(
        0.01,
        0.01,
        '$\\theta$ = %.5f +/- %.5f mas' % (fit['p'][3], fit['dp'][3]),
        ha='left',
        va='bottom',
        transform=ax[0, 0].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0, 0].set_ylim(temp)
    ax[0, 0].set_ylabel('Data $|V|^2$')
    ax[0, 0].legend(loc='upper left')
    ax[1, 0].plot(v2_mod, v2_res / dv2 / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1, 0].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1, 0].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1, 0].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1, 0].get_ylim()))
    ax[1, 0].set_ylim(-ylim, ylim)
    ax[1, 0].set_xlabel('Model $|V|^2$')
    ax[1, 0].set_ylabel('Res. [$\sigma$/$\chi$]')
    ax[0, 1].errorbar(
        cp_mod, cp, yerr=dcp, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0, 1].plot(
        [np.min(cp_mod), np.max(cp_mod)], [np.min(cp_mod), np.max(cp_mod)], color=modelcol, zorder=4, label='Model'
    )
    ax[0, 1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[0, 1].text(
        0.01,
        0.01,
        '$f$ = %.3f +/- %.3f %%' % (fit['p'][0] * 100.0, fit['dp'][0] * 100.0),
        ha='left',
        va='bottom',
        transform=ax[0, 1].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0, 1].set_ylabel('Data closure phase [rad]')
    ax[0, 1].legend(loc='upper left')
    ax[1, 1].plot(cp_mod, cp_res / dcp / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1, 1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1, 1].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1, 1].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1, 1].get_ylim()))
    ax[1, 1].set_ylim(-ylim, ylim)
    ax[1, 1].set_xlabel('Model closure phase [rad]')
    ax[1, 1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=1.0 / 3.0, hspace=0.0)
    fig.align_ylabels()
    plt.suptitle('Uniform disk with point-source companion fit')
    util.save_ofile(ofile, 'v2_cp_ud_bin')
    # plt.show()
    plt.close()


def kp_bin(data_list, fit, smear=None, ofile=None):
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
        rho = np.sqrt(dra**2 + ddec**2)
        phi = np.rad2deg(np.arctan2(dra, ddec))
        if pa_mtoc == '-':
            phi -= data_list[i]['pa']
        elif pa_mtoc == '+':
            phi += data_list[i]['pa']
        else:
            raise UserWarning('Model to chip conversion for position angle not known')
        phi = ((phi + 180.0) % 360.0) - 180.0
        dra_temp = rho * np.sin(np.deg2rad(phi))
        ddec_temp = rho * np.cos(np.deg2rad(phi))
        p0_temp = np.array([fit['p'][0].copy(), dra_temp, ddec_temp])
        kp += [data_list[i]['kp'].flatten()]
        dkp += [data_list[i]['dkp'].flatten()]
        vis_mod = util.vis_bin(p0=p0_temp, data=data_list[i], smear=smear)
        kp_mod += [util.v2kp(vis_mod, data=data_list[i]).flatten()]
    kp = np.concatenate(kp)
    dkp = np.concatenate(dkp)
    kp_mod = np.concatenate(kp_mod)
    kp_res = kp - kp_mod

    fig, ax = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(6.4, 4.8))
    ax[0].errorbar(
        kp_mod, kp, yerr=dkp, elinewidth=1, ls='none', marker='s', ms=2, color=datacol, zorder=1, label='Data'
    )
    ax[0].plot(
        [np.min(kp_mod), np.max(kp_mod)], [np.min(kp_mod), np.max(kp_mod)], color=modelcol, zorder=4, label='Model'
    )
    ax[0].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[0].text(
        0.01,
        0.01,
        '$f$ = %.3f +/- %.3f %%' % (fit['p'][0] * 100.0, fit['dp'][0] * 100.0),
        ha='left',
        va='bottom',
        transform=ax[0].transAxes,
        zorder=5,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[0].set_ylabel('Data kernel phase [rad]')
    ax[0].legend(loc='upper left')
    ax[1].plot(kp_mod, kp_res / dkp / np.sqrt(fit['chi2_red']), ls='none', marker='s', ms=2, color=datacol, zorder=1)
    ax[1].axhline(0.0, ls='--', color=gridcol, zorder=2)
    text = ax[1].text(
        0.99, 0.96, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='top', transform=ax[1].transAxes, zorder=3
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ylim = np.max(np.abs(ax[1].get_ylim()))
    ax[1].set_ylim(-ylim, ylim)
    ax[1].set_xlabel('Model kernel phase [rad]')
    ax[1].set_ylabel('Res. [$\sigma$/$\chi$]')
    plt.subplots_adjust(wspace=0.25, hspace=0.0)
    fig.align_ylabels()
    plt.suptitle('Point-source companion fit')
    util.save_ofile(ofile, 'kp_bin')
    # plt.show()
    plt.close()


def lincmap(
    pps,
    pes,
    chi2s,
    nsigmas,
    fit,
    sep_range,
    step_size,
    vmin=None,
    vmax=None,
    ofile=None,
    searchbox=None,
    plot_nsigma=False,
):
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
        Array of shape (RA steps x DEC steps) containing best fit
        chi-squared for the grid.
    nsigmas: array
        Array of shape (RA steps x DEC steps) containing best fit detection
        significance for the grid.
    fit: dict
        Model fit whose chi-squared map shall be plotted.
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    vmin : float
        Log10 of contrast map vmin.
    vmax : float
        Log10 of contrast map vmax.
    ofile: str
        Path under which figures shall be saved.
    searchbox: dict
        Search box inside of which the companion is expected to be. Accepted
        formats are {'RA': [RA_min, RA_max], 'DEC': [DEC_min, DEC_max], 'rho':
        [rho_min, rho_max], 'phi': [phi_min, phi_max]}. Note that -180 <= phi
        < 180.
    plot_nsigma: bool
            Plot detection significance instead of chi-squared map.
    """

    grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range, step_size=step_size, verbose=False)
    emax = np.nanmax(grid_ra_dec[0])
    sep = np.sqrt(fit['p'][1] ** 2 + fit['p'][2] ** 2)
    pa = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
    rad, avg = ot.azimuthalAverage(np.abs(pps[0]), returnradii=True, binsize=1.0)
    std = ot.azimuthalAverage(np.abs(pps[0]), binsize=1.0, stddev=True)
    rad *= step_size
    if searchbox is not None:
        searchmap = np.ones_like(grid_ra_dec[0].flatten())
        if 'RA' in searchbox.keys():
            RA = grid_ra_dec[0].flatten()
            searchmap[RA < searchbox['RA'][0]] = 0.0
            searchmap[RA > searchbox['RA'][1]] = 0.0
        if 'DEC' in searchbox.keys():
            DEC = grid_ra_dec[1].flatten()
            searchmap[DEC < searchbox['DEC'][0]] = 0.0
            searchmap[DEC > searchbox['DEC'][1]] = 0.0
        if 'rho' in searchbox.keys():
            rho = np.sqrt(grid_ra_dec[0].flatten() ** 2 + grid_ra_dec[1].flatten() ** 2)
            searchmap[rho < searchbox['rho'][0]] = 0.0
            searchmap[rho > searchbox['rho'][1]] = 0.0
        if 'phi' in searchbox.keys():
            phi = np.rad2deg(np.arctan2(grid_ra_dec[0].flatten(), grid_ra_dec[1].flatten()))
            searchmap[phi < searchbox['phi'][0]] = 0.0
            searchmap[phi > searchbox['phi'][1]] = 0.0
        searchmap = searchmap.reshape(grid_ra_dec[0].shape)
        searchmap[np.isnan(grid_ra_dec[0])] = 0.0

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(1, 3, figsize=(19.2, 4.8))
    temp = pps[0]
    temp[temp <= 0.0] = np.min(temp[temp > 0.0])
    if vmin is None:
        vmin = np.log10(np.nanpercentile(temp, 55.0))
    if vmax is None:
        vmax = -1
    p0 = ax[0].imshow(
        np.log10(temp),
        cmap='hot',
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
    )
    c0 = plt.colorbar(p0, ax=ax[0])
    c0.set_label(r'$\mathrm{log_{10}}$(relative flux)', rotation=270, labelpad=20)
    ax[0].plot(0.0, 0.0, marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='white', lw=5, fill=False)
    ax[0].add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='black', lw=2.5, fill=False)
    ax[0].add_artist(cc)
    ax[0].set_xlabel('$\Delta$RA [mas]')
    ax[0].set_ylabel('$\Delta$DEC [mas]')
    ax[0].set_title('Linear contrast map')
    if not plot_nsigma:
        p1 = ax[1].imshow(
            chi2s / fit['ndof'],
            cmap='cubehelix',
            origin='lower',
            extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
        )
    else:
        p1 = ax[1].imshow(
            nsigmas,
            cmap='cubehelix_r',
            origin='lower',
            extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
        )
    c1 = plt.colorbar(p1, ax=ax[1])
    if not plot_nsigma:
        c1.set_label('$\chi^2$', rotation=270, labelpad=20)
    else:
        c1.set_label('$N_{\sigma}$', rotation=270, labelpad=20)
    if searchbox is not None:
        ax[1].imshow(
            searchmap,
            cmap='Reds',
            origin='lower',
            extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
            alpha=0.5,
        )
    ax[1].plot(0.0, 0.0, marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='white', lw=5, fill=False)
    ax[1].add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='black', lw=2.5, fill=False)
    ax[1].add_artist(cc)
    text = ax[1].text(
        0.01,
        0.99,
        '$f$ = %.3e +/- %.3e %%' % (fit['p'][0] * 100.0, fit['dp'][0] * 100.0),
        ha='left',
        va='top',
        color='black',
        transform=ax[1].transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[1].text(
        0.01, 0.06, '$\\rho$ = %.1f mas' % sep, ha='left', va='bottom', color='black', transform=ax[1].transAxes
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[1].text(
        0.01, 0.01, '$\\varphi$ = %.1f deg' % pa, ha='left', va='bottom', color='black', transform=ax[1].transAxes
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[1].text(
        0.99,
        0.99,
        '$N_{\sigma}$ = %.1f' % fit['nsigma'],
        ha='right',
        va='top',
        color='black',
        transform=ax[1].transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax[1].text(
        0.99,
        0.01,
        '$\chi^2$ = %.3f' % fit['chi2_red'],
        ha='right',
        va='bottom',
        color='black',
        transform=ax[1].transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax[1].set_xlabel('$\Delta$RA [mas]')
    ax[1].set_ylabel('$\Delta$DEC [mas]')
    if not plot_nsigma:
        if searchbox is None:
            ax[1].set_title('Chi-squared map')
        else:
            ax[1].set_title('Chi-squared map (search region shaded red)')
    else:
        if searchbox is None:
            ax[1].set_title('Detection significance')
        else:
            ax[1].set_title('Detection significance (search region shaded red)')
    ax[2].plot(rad, avg, color=colors[0], label='avg')
    ax[2].fill_between(rad, avg - std, avg + std, ec='None', fc=colors[0], alpha=1.0 / 3.0)
    ax[2].grid(axis='y')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('Separation [mas]')
    ax[2].set_ylabel('Contrast')
    ax[2].set_title('Detection limits (1-$\sigma$)')
    ax[2].legend(loc='upper right')
    # ax = ax[2].twinx()
    # ax.plot(rad, max/avg, color='black', ls=':')
    # ax.set_ylabel(r'Significance [$\sigma$]', rotation=270, labelpad=20)
    plt.tight_layout()
    util.save_ofile(ofile, 'lincmap')
    # plt.show()
    plt.close()


def chi2map(pps_unique, chi2s_unique, fit, sep_range, step_size, ofile=None, searchbox=None):
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
    searchbox: dict
        Search box inside of which the companion is expected to be. Accepted
        formats are {'RA': [RA_min, RA_max], 'DEC': [DEC_min, DEC_max], 'rho':
        [rho_min, rho_max], 'phi': [phi_min, phi_max]}. Note that -180 <= phi
        < 180.
    """

    grid_ra_dec_fine, grid_sep_pa_fine = util.get_grid(sep_range=sep_range, step_size=step_size / 4.0, verbose=False)
    emax = np.nanmax(grid_ra_dec_fine[0])
    func = Rbf(pps_unique[:, 1], pps_unique[:, 2], chi2s_unique, function='linear')
    chi2s_rbf = func(grid_ra_dec_fine[0].flatten(), grid_ra_dec_fine[1].flatten()).reshape(grid_ra_dec_fine[0].shape)
    sep = np.sqrt(fit['p'][1] ** 2 + fit['p'][2] ** 2)
    pa = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
    dsep = np.sqrt((fit['p'][1] / sep * fit['dp'][1]) ** 2 + (fit['p'][2] / sep * fit['dp'][2]) ** 2)
    dpa = np.rad2deg(np.sqrt((fit['p'][2] / sep**2 * fit['dp'][1]) ** 2 + (-fit['p'][1] / sep**2 * fit['dp'][2]) ** 2))
    if searchbox is not None:
        searchmap = np.ones_like(grid_ra_dec_fine[0].flatten())
        if 'RA' in searchbox.keys():
            RA = grid_ra_dec_fine[0].flatten()
            searchmap[RA < searchbox['RA'][0]] = 0.0
            searchmap[RA > searchbox['RA'][1]] = 0.0
        if 'DEC' in searchbox.keys():
            DEC = grid_ra_dec_fine[1].flatten()
            searchmap[DEC < searchbox['DEC'][0]] = 0.0
            searchmap[DEC > searchbox['DEC'][1]] = 0.0
        if 'rho' in searchbox.keys():
            rho = np.sqrt(grid_ra_dec_fine[0].flatten() ** 2 + grid_ra_dec_fine[1].flatten() ** 2)
            searchmap[rho < searchbox['rho'][0]] = 0.0
            searchmap[rho > searchbox['rho'][1]] = 0.0
        if 'phi' in searchbox.keys():
            phi = np.rad2deg(np.arctan2(grid_ra_dec_fine[0].flatten(), grid_ra_dec_fine[1].flatten()))
            searchmap[phi < searchbox['phi'][0]] = 0.0
            searchmap[phi > searchbox['phi'][1]] = 0.0
        searchmap = searchmap.reshape(grid_ra_dec_fine[0].shape)
        searchmap[np.isnan(grid_ra_dec_fine[0])] = 0.0

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    p0 = ax.imshow(
        chi2s_rbf / fit['ndof'],
        cmap='cubehelix',
        origin='lower',
        extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
    )
    c0 = plt.colorbar(p0, ax=ax)
    c0.set_label('$\chi^2$', rotation=270, labelpad=20)
    if searchbox is not None:
        ax.imshow(
            searchmap,
            cmap='Reds',
            origin='lower',
            extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
            alpha=0.5,
        )
    ax.plot(0.0, 0.0, marker='*', color='black', markersize=10)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='white', lw=5, fill=False)
    ax.add_artist(cc)
    cc = plt.Circle((fit['p'][1], fit['p'][2]), emax / 10.0, color='black', lw=2.5, fill=False)
    ax.add_artist(cc)
    text = ax.text(
        0.01,
        0.99,
        '$f$ = %.3f +/- %.3f %%' % (fit['p'][0] * 100.0, fit['dp'][0] * 100.0),
        ha='left',
        va='top',
        color='black',
        transform=ax.transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(
        0.01,
        0.06,
        '$\\rho$ = %.1f +/- %.1f mas' % (sep, dsep),
        ha='left',
        va='bottom',
        color='black',
        transform=ax.transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(
        0.01,
        0.01,
        '$\\varphi$ = %.1f +/- %.1f deg' % (pa, dpa),
        ha='left',
        va='bottom',
        color='black',
        transform=ax.transAxes,
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(
        0.99, 0.99, '$N_{\sigma}$ = %.1f' % fit['nsigma'], ha='right', va='top', color='black', transform=ax.transAxes
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    text = ax.text(
        0.99, 0.01, '$\chi^2$ = %.3f' % fit['chi2_red'], ha='right', va='bottom', color='black', transform=ax.transAxes
    )
    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    ax.set_xlabel('$\Delta$RA [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    if searchbox is None:
        plt.suptitle('Chi-squared map')
    else:
        plt.suptitle('Chi-squared map (search region shaded red)')
    util.save_ofile(ofile, 'chi2map')
    # plt.show()
    plt.close()

    return chi2s_rbf, grid_ra_dec_fine


def chains(fit, samples, ofile=None, fixpos=False):
    """
    Parameters
    ----------
    fit: dict
        Model fit whose MCMC chains shall be plotted.
    samples: array
        Posterior samples.
    ofile: str
        Path under which figures shall be saved.
    fixpos: bool
        Fix position of fit?
    """

    if fit['model'] == 'ud':
        fig = plt.figure()
        plt.plot(samples[:, 0], color=datacol)
        plt.axhline(np.percentile(samples[:, 0], 50.0), color=modelcol, label='MCMC median')
        plt.axhline(fit['p'][0], ls='--', color=gridcol, label='Initial guess')
        plt.xlabel('Step')
        plt.ylabel('$\\theta$ [mas]')
        plt.legend(loc='upper right')
        plt.suptitle('MCMC chains')
        util.save_ofile(ofile, 'mcmc_chains')
        # plt.show()
        plt.close()
    elif (fit['model'] == 'bin'):
        if fixpos:
            ylabels = ['$f$ [%]']
            fig = plt.figure(figsize=(6.4, 1.6))
            plt.plot(samples[:, 0] * 100.0, color=datacol)
            plt.axhline(np.percentile(samples[:, 0], 50.0) * 100.0, color=modelcol, label='MCMC median')
            plt.axhline(fit['p'][0] * 100.0, ls='--', color=gridcol, label='Initial guess')
            plt.xlabel('Step')
            plt.ylabel(ylabels[0])
            plt.tight_layout()
        if (len(fit['p']) == 3):
            ylabels = ['$f$ [%]', '$\\rho$ [mas]', '$\\varphi$ [deg]']
            rho = np.sqrt(samples[:, 1]**2+samples[:, 2]**2)
            rho0 = np.sqrt(fit['p'][1]**2+fit['p'][2]**2)
            phi = np.rad2deg(np.arctan2(samples[:, 1], samples[:, 2]))
            phi0 = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
            fig, ax = plt.subplots(len(fit['p']), 1, sharex='col', figsize=(6.4, 1.6*len(fit['p'])))
            ax[0].plot(samples[:, 0]*100., color=datacol)
            ax[0].axhline(np.percentile(samples[:, 0], 50.)*100., color=modelcol, label='MCMC median')
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
        else:
            ylabels = ['$f%.0f$ [%%]' % (i + 1) for i in range(len(fit['p']) - 2)]
            ylabels += ['$\\rho$ [mas]', '$\\varphi$ [deg]']
            rho = np.sqrt(samples[:, -2]**2+samples[:, -1]**2)
            rho0 = np.sqrt(fit['p'][-2]**2+fit['p'][-1]**2)
            phi = np.rad2deg(np.arctan2(samples[:, -2], samples[:, -1]))
            phi0 = np.rad2deg(np.arctan2(fit['p'][-2], fit['p'][-1]))
            fig, ax = plt.subplots(len(fit['p']), 1, sharex='col', figsize=(6.4, 1.6*len(fit['p'])))
            for i in range(len(fit['p']) - 2):
                ax[i].plot(samples[:, i]*100., color=datacol)
                ax[i].axhline(np.percentile(samples[:, i], 50.)*100., color=modelcol, label='MCMC median')
                ax[i].axhline(fit['p'][i]*100., ls='--', color=gridcol, label='Initial guess')
            ax[-2].plot(rho, color=datacol)
            ax[-2].axhline(np.percentile(rho, 50.), color=modelcol)
            ax[-2].axhline(rho0, ls='--', color=gridcol)
            ax[-1].plot(phi, color=datacol)
            ax[-1].axhline(np.percentile(phi, 50.), color=modelcol)
            ax[-1].axhline(phi0, ls='--', color=gridcol)
            plt.xlabel('Step')
            for i in range(len(fit['p'])):
                ax[i].set_ylabel(ylabels[i])
                if (i == 0):
                    ax[i].legend(loc='upper right')
            plt.subplots_adjust(wspace=0.25, hspace=0.)
            fig.align_ylabels()
        plt.suptitle('MCMC chains')
        util.save_ofile(ofile, 'mcmc_chains')
        # plt.show()
        plt.close()
    else:
        ylabels = ['$f$ [%]', '$\\rho$ [mas]', '$\\varphi$ [deg]', '$\\theta$ [mas]']
        rho = np.sqrt(samples[:, 1] ** 2 + samples[:, 2] ** 2)
        rho0 = np.sqrt(fit['p'][1] ** 2 + fit['p'][2] ** 2)
        phi = np.rad2deg(np.arctan2(samples[:, 1], samples[:, 2]))
        phi0 = np.rad2deg(np.arctan2(fit['p'][1], fit['p'][2]))
        fig, ax = plt.subplots(len(fit['p']), 1, sharex='col', figsize=(6.4, 1.6 * len(fit['p'])))
        ax[0].plot(samples[:, 0] * 100.0, color=datacol)
        ax[0].axhline(np.percentile(samples[:, 0], 50.0) * 100.0, color=modelcol, label='MCMC median')
        ax[0].axhline(fit['p'][0] * 100.0, ls='--', color=gridcol, label='Initial guess')
        ax[1].plot(rho, color=datacol)
        ax[1].axhline(np.percentile(rho, 50.0), color=modelcol)
        ax[1].axhline(rho0, ls='--', color=gridcol)
        ax[2].plot(phi, color=datacol)
        ax[2].axhline(np.percentile(phi, 50.0), color=modelcol)
        ax[2].axhline(phi0, ls='--', color=gridcol)
        ax[3].plot(samples[:, 3], color=datacol)
        ax[3].axhline(np.percentile(samples[:, 3], 50.0), color=modelcol)
        ax[3].axhline(fit['p'][3], ls='--', color=gridcol)
        plt.xlabel('Step')
        for i in range(len(fit['p'])):
            ax[i].set_ylabel(ylabels[i])
            if i == 0:
                ax[i].legend(loc='upper right')
        plt.subplots_adjust(wspace=0.25, hspace=0.0)
        fig.align_ylabels()
        plt.suptitle('MCMC chains')
        util.save_ofile(ofile, 'mcmc_chains')
        # plt.show()
        plt.close()


def corner(fit, samples, ofile=None, fixpos=False):
    """
    Parameters
    ----------
    fit: dict
        Model fit whose posterior distribution shall be plotted.
    samples: array
        Posterior samples.
    ofile: str
        Path under which figures shall be saved.
    fixpos: bool
        Fix position of fit?
    """
    if fit['model'] == 'ud':
        fig = cp.corner(
            samples,
            labels=[r'$\theta$ [mas]'],
            titles=[r'$\theta$'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
        )
        util.save_ofile(ofile, 'mcmc_corner')
        # plt.show()
        plt.close()
    elif fit['model'] == 'bin':
        if fixpos:
            temp = samples.copy()
            temp[:, 0] *= 100.
            fig = cp.corner(
                temp,
                labels=[r'$f$ [%]'],
                titles=[r'$f$'],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt='.3f',
            )
            util.save_ofile(ofile, 'mcmc_corner')
            # plt.show()
            plt.close()
        else:
            if samples.shape[1] > 3:
                temp = samples.copy()
                temp[:, :-2] *= 100.
                temp[:, -2] = np.sqrt(samples[:, -2]**2+samples[:, -1]**2)
                temp[:, -1] = np.rad2deg(np.arctan2(samples[:, -2], samples[:, -1]))
                fig = cp.corner(
                    temp,
                    labels=[r'$f%.0f$ [%%]' % (i + 1) for i in range(temp.shape[1] - 2)]
                    + [r'$\rho$ [mas]', r'$\varphi$ [deg]'],
                    titles=[r'$f%.0f$' % (i + 1) for i in range(temp.shape[1] - 2)] + [r'$\rho$', r'$\varphi$'],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt='.3f',
                )
                util.save_ofile(ofile, 'mcmc_corner')
                # plt.show()
                plt.close()
            else:
                temp = samples.copy()
                temp[:, 0] *= 100.0
                temp[:, 1] = np.sqrt(samples[:, 1] ** 2 + samples[:, 2] ** 2)
                temp[:, 2] = np.rad2deg(np.arctan2(samples[:, 1], samples[:, 2]))
                fig = cp.corner(
                    temp,
                    labels=[r'$f$ [%]', r'$\rho$ [mas]', r'$\varphi$ [deg]'],
                    titles=[r'$f$', r'$\rho$', r'$\varphi$'],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt='.3f',
                )
                util.save_ofile(ofile, 'mcmc_corner')
                # plt.show()
                plt.close()
    else:
        temp = samples.copy()
        temp[:, 0] *= 100.0
        temp[:, 1] = np.sqrt(samples[:, 1] ** 2 + samples[:, 2] ** 2)
        temp[:, 2] = np.rad2deg(np.arctan2(samples[:, 1], samples[:, 2]))
        fig = cp.corner(
            temp,
            labels=[r'$f$ [%]', r'$\rho$ [mas]', r'$\varphi$ [deg]', r'$\theta$ [mas]'],
            titles=[r'$f$', r'$\rho$', r'$\varphi$', r'$\theta$'],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
        )
        util.save_ofile(ofile, 'mcmc_corner')
        # plt.show()
        plt.close()


def detlim(ffs_absil, ffs_injection, sigma, sep_range, step_size, ofile=None):
    """
    Parameters
    ----------
    ffs_absil: array
        2D detection limit map from Absil method.
    ffs_injection: array
        2D detection limit map from Injection method.
    sigma: int
        Confidence level for which the detection limits shall be computed.
    sep_range: tuple of float
        Min. and max. angular separation of grid (mas).
    step_size: float
        Step size of grid (mas).
    ofile: str
        Path under which figures shall be saved.
    """

    grid_ra_dec, grid_sep_pa = util.get_grid(sep_range=sep_range, step_size=step_size, verbose=False)
    emax = np.nanmax(grid_ra_dec[0])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(2 * 6.4, 2 * 4.8))
    ax = plt.subplot(gs[0, 0])
    # p0 = ax.imshow(-2.5*np.log10(ffs_absil), origin='lower', cmap='cubehelix', extent=(emax+step_size/2., -emax-step_size/2., -emax-step_size/2., emax+step_size/2.), vmin=3.8, vmax=5.1)
    p0 = ax.imshow(
        -2.5 * np.log10(ffs_absil),
        origin='lower',
        cmap='cubehelix',
        extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
    )
    c0 = plt.colorbar(p0, ax=ax)
    c0.set_label('Contrast [mag]', rotation=270, labelpad=20)
    ax.plot(0.0, 0.0, marker='*', color='black', markersize=10)
    ax.set_xlabel('$\Delta$RA [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    ax.set_title('Method Absil')
    ax = plt.subplot(gs[0, 1])
    # p1 = ax.imshow(-2.5*np.log10(ffs_injection), origin='lower', cmap='cubehelix', extent=(emax+step_size/2., -emax-step_size/2., -emax-step_size/2., emax+step_size/2.), vmin=3.5, vmax=5.7)
    p1 = ax.imshow(
        -2.5 * np.log10(ffs_injection),
        origin='lower',
        cmap='cubehelix',
        extent=(emax + step_size / 2.0, -emax - step_size / 2.0, -emax - step_size / 2.0, emax + step_size / 2.0),
    )
    c1 = plt.colorbar(p1, ax=ax)
    c1.set_label('Contrast [mag]', rotation=270, labelpad=20)
    ax.plot(0.0, 0.0, marker='*', color='black', markersize=10)
    ax.set_xlabel('$\Delta$RA [mas]')
    ax.set_ylabel('$\Delta$DEC [mas]')
    ax.set_title('Method Injection')
    ax = plt.subplot(gs[1, :])

    # temp = np.load('/Users/jkammerer/Documents/Code/fouriever/test/axcir_smear_nocov_sub_detlim_absil.npy', allow_pickle=True)
    # ax.plot(temp[0], temp[1], color=colors[0], lw=3, label='Method Absil (w/o cov)')
    # temp = np.load('/Users/jkammerer/Documents/Code/fouriever/test/axcir_smear_nocov_sub_detlim_injection.npy', allow_pickle=True)
    # ax.plot(temp[0], temp[1], color=colors[1], lw=3, label='Method Injection (w/o cov)')

    rad, avg = ot.azimuthalAverage(ffs_absil, returnradii=True, binsize=1)
    ax.plot(rad * step_size, -2.5 * np.log10(avg), color=colors[0], lw=3, label='Method Absil')
    # ax.plot(rad*step_size, -2.5*np.log10(avg), color=colors[0], lw=3, ls='--', label='Method Absil (w/ cov)')
    data = []
    data += [rad * step_size]  # mas
    data += [-2.5 * np.log10(avg)]  # mag
    data = np.array(data)
    util.save_ofile(ofile, 'detlim_absil', data, out_ext='npy')
    rad, avg = ot.azimuthalAverage(ffs_injection, returnradii=True, binsize=1)
    ax.plot(rad * step_size, -2.5 * np.log10(avg), color=colors[1], lw=3, label='Method Injection')
    # ax.plot(rad*step_size, -2.5*np.log10(avg), color=colors[1], lw=3, ls='--', label='Method Injection (w/ cov)')
    data = []
    data += [rad * step_size]  # mas
    data += [-2.5 * np.log10(avg)]  # mag
    data = np.array(data)
    util.save_ofile(ofile, 'detlim_injection', data, out_ext='npy')

    # temp_X = np.load('/Users/jkammerer/Documents/Code/fouriever/test/Absil_X.npy')
    # temp_Y = np.load('/Users/jkammerer/Documents/Code/fouriever/test/Absil_Y.npy')
    # temp_f = np.load('/Users/jkammerer/Documents/Code/fouriever/test/Absil_f.npy')
    # step_size = np.abs(temp_X[0, 0]-temp_X[0, 1])
    # rad, avg = ot.azimuthalAverage(temp_f, returnradii=True, binsize=1)
    # rad = rad*step_size
    # ww = rad <= 40.
    # ax.plot(rad[ww], avg[ww], color=colors[0], lw=3, alpha=1./3., label='Method Absil (CANDID)')
    # temp_X = np.load('/Users/jkammerer/Documents/Code/fouriever/test/injection_X.npy')
    # temp_Y = np.load('/Users/jkammerer/Documents/Code/fouriever/test/injection_Y.npy')
    # temp_f = np.load('/Users/jkammerer/Documents/Code/fouriever/test/injection_f.npy')
    # step_size = np.abs(temp_X[0, 0]-temp_X[0, 1])
    # rad, avg = ot.azimuthalAverage(temp_f, returnradii=True, binsize=1)
    # rad = rad*step_size
    # ww = rad <= 40.
    # ax.plot(rad[ww], avg[ww], color=colors[1], lw=3, alpha=1./3., label='Method Injection (CANDID)')

    ax.grid(axis='both')
    ax.invert_yaxis()
    # ax.set_xlim([0., 60.])
    ax.set_xlabel('Separation [mas]')
    ax.set_ylabel('Contrast [mag]')
    ax.legend(loc='upper right')
    plt.suptitle('Detection limits (' + str(sigma) + '-$\sigma$)')
    plt.tight_layout()
    util.save_ofile(ofile, 'detlim')
    # plt.show()
    plt.close()
