"""A selection of useful functions for optics, especially Fourier optics. The
documentation is designed to be used with sphinx (still lots to do)

Authors:
Dr Michael Ireland
Adam Rains
"""

from __future__ import print_function, division
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import optimize
from .utils import *
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(1.0)
    nthreads=6 
except:
    nthreads=0

#On load, create a quick index of the first 100 Zernike polynomials, according to OSA/ANSI:
MAX_ZERNIKE=105
ZERNIKE_N = np.empty(MAX_ZERNIKE, dtype=int)
ZERNIKE_M = np.empty(MAX_ZERNIKE, dtype=int)
ZERNIKE_NORM = np.ones(MAX_ZERNIKE)
n=0
m=0
for z_ix in range(0,MAX_ZERNIKE):
    ZERNIKE_N[z_ix] = n
    ZERNIKE_M[z_ix] = m
    if m==0:
        ZERNIKE_NORM[z_ix] = np.sqrt(n+1)
    else:
        ZERNIKE_NORM[z_ix] = np.sqrt(2*(n+1))
    if m==n:
        n += 1
        m = -n
    else:
        m += 2


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, 
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None, return_max=False):
    """
    Calculate the azimuthally averaged radial profile.
    NB: This was found online and should be properly credited! Modified by MJI

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    return_max - (MJI) Return the maximum index.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # Find out which radial bin each point in the map belongs to
    whichbin = np.digitize(r.flat,bins)

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    nr = np.bincount(whichbin)[1:]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape

    if stddev:
        radial_prof = np.array([image.flat[whichbin==b].std() for b in range(1,nbins+1)])
    elif return_max:
        radial_prof = np.array([np.append((image*weights).flat[whichbin==b],-np.inf).max() for b in range(1,nbins+1)])
    else:
        radial_prof = np.array([(image*weights).flat[whichbin==b].sum() / weights.flat[whichbin==b].sum() for b in range(1,nbins+1)])

    #import pdb; pdb.set_trace()

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof

def propagate_by_fresnel(wf, m_per_pix, d, wave):
    """Propagate a wave by Fresnel diffraction
    
    Parameters
    ----------
    wf: float array
        Wavefront, i.e. a complex electric field in the scalar approximation.
    m_per_pix: float
        Scale of the pixels in the input wavefront in metres.
    d: float
        Distance to propagate the wavefront.
    wave: float
        Wavelength in metres.
        
    Returns
    -------
    wf_new: float array
        Wavefront after propagating.
    """
    #Notation on Mike's board
    sz = wf.shape[0]
    if (wf.shape[0] != wf.shape[1]):
        print("ERROR: Input wavefront must be square")
        raise UserWarning
    
    #The code below came from the board, i.e. via Huygen's principle.
    #We got all mixed up when converting to Fourier transform co-ordinates.
    #Co-ordinate axis of the wavefront. Not that 0 must be in the corner.
    #x = (((np.arange(sz)+sz/2) % sz) - sz/2)*m_per_pix
    #xy = np.meshgrid(x,x)
    #rr =np.sqrt(xy[0]**2 + xy[1]**2)
    #h_func = np.exp(1j*np.pi*rr**2/wave/d)
    #h_ft = np.fft.fft2(h_func)
    
    #Co-ordinate axis of the wavefront Fourier transform. Not that 0 must be in the corner.
    #x is in cycles per wavefront dimension.
    x = (((np.arange(sz)+sz/2) % sz) - sz/2)/m_per_pix/sz
    xy = np.meshgrid(x,x)
    uu =np.sqrt(xy[0]**2 + xy[1]**2)
    h_ft = np.exp(1j*np.pi*uu**2*wave*d)
    
    g_ft = np.fft.fft2(np.fft.fftshift(wf))*h_ft
    wf_new = np.fft.ifft2(g_ft)
    return np.fft.fftshift(wf_new)

def airy(x_in, obstruction_sz=0):
    """Return an Airy disk as an electric field as a function of angle in units of 
    lambda/D, with the possibility of a circular obstruction that is a fraction of the
    pupil size.
    
    The total intensity is proportional to the area, so the peak intensity is 
    proportional to the square of the area, and the peak electric field proportional
    to the area.
    
    Parameters
    ----------
    x: array-like 
        Angular position in units of lambda/D for the Airy function.
        
    obstruction_sz: float
        Fractional size of the obstruction for a circular aperture.
    """
    
    #Implicitly do a shallow copy of x_in to a numpy array x.
    if type(x_in)==int or type(x_in)==float:
        x = np.array([x_in])
    else:
        try:
            x = np.array(x_in).flatten()
        except:
            print("ERROR: x must be castable to an array")
            raise UserWarning
        
    ix = np.where(x>0)[0]
    y1 = np.ones(x.shape)
    y1[ix] = 2*special.jn(1,np.pi*x[ix])/(np.pi*x[ix])
    
    if obstruction_sz>0:
        y2 = np.ones(x.shape)
        y2[ix] = 2*special.jn(1,np.pi*x[ix]*obstruction_sz)/(np.pi*x[ix]*obstruction_sz)
        y1 -= obstruction_sz**2 * y2
        y1 /= (1 - obstruction_sz**2)

    #Return the same data type input (within reason):
    if type(x_in)==int or type(x_in)==float:
    	y1 = y1[0]
    elif type(x_in)==list:
        y1 = list(y1)
    else:
        y1 = y1.reshape(np.array(x_in).shape)

    return y1

def curved_wf(sz,m_per_pix,f_length=np.infty,wave=633e-9, tilt=[0.0,0.0], power=None,diam=None,defocus=None):
    """A curved wavefront centered on the *middle*
    of the python array.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    m_per_pix: float
        Meters per pixel
    tilt: float (optional)
        Tilt of the wavefront in radians in the x and y directions. 
    wave: float
        Wavelength in m       
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    rr =np.sqrt(xy[0]**2 + xy[1]**2)
    if not power:
        power = 1.0/f_length
    if not diam:
        diam=sz*m_per_pix
    #The following line computes phase in *wavelengths*
    if defocus:
        phase = defocus*(rr*m_per_pix/diam*2)**2
    else:
        phase = 0.5*m_per_pix**2/wave*power*rr**2 
    phase += tilt[0]*xy[0]*diam/sz/wave
    phase += tilt[1]*xy[1]*diam/sz/wave

    return np.exp(2j*np.pi*phase)

def zernike(sz, coeffs=[0.,0.,0.], diam=None, rms_norm=False):
    """A zernike wavefront centered on the *middle*
    of the python array.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    coeffs: float array
        Zernike coefficients, starting with piston.
    diam: float
        Diameter for normalisation in pixels.      
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    if not diam:
        diam=sz
    rr = np.sqrt(xy[0]**2 + xy[1]**2)/(diam/2)
    phi = np.arctan2(xy[0], xy[1])
    n_coeff = len(coeffs)
    phase = np.zeros((sz,sz))
    #Loop over each zernike term.
    for coeff,n,m_signed,norm in zip(coeffs,ZERNIKE_N[:n_coeff], ZERNIKE_M[:n_coeff], ZERNIKE_NORM[:n_coeff]):
        m = np.abs(m_signed)
        #Reset the term.
        term = np.zeros((sz,sz))
        
        #The "+1" is to make an inclusive range.
        for k in range(0,(n-m)//2+1):
            term += (-1)**k * np.math.factorial(n-k) / np.math.factorial(k)/\
                np.math.factorial((n+m)/2-k) / np.math.factorial((n-m)/2-k) *\
                rr**(n-2*k)
        if m_signed < 0:
            term *= np.sin(m*phi)
        if m_signed > 0:
            term *= np.cos(m*phi)
            
        #Add to the phase
        if rms_norm:
            phase += term*coeff*norm
        else:
            phase += term*coeff

    return phase

def zernike_wf(sz, coeffs=[0.,0.,0.], diam=None, rms_norm=False):
    """A zernike wavefront centered on the *middle*
    of the python array. Amplitude of coefficients
    normalised in radians.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    coeffs: float array
        Zernike coefficients, starting with piston.
    diam: float
        Diameter for normalisation in pixels.      
    """
    return np.exp(1j*zernike(sz, coeffs, diam, rms_norm))

def zernike_amp(sz, coeffs=[0.,0.,0.], diam=None, rms_norm=False):
    """A zernike based amplitude centered on the *middle*
    of the python array.
    
    Parameters
    ----------
    sz: int
        Size of the wavefront in pixels
    coeffs: float array
        Zernike coefficients, starting with piston.
    diam: float
        Diameter for normalisation in pixels.      
    """
    return np.exp(zernike(sz, coeffs, diam, rms_norm))

def pd_images(foc_offsets=[0,0], xt_offsets = [0,0], yt_offsets = [0,0], 
    phase_zernikes=[0,0,0,0], amp_zernikes = [0], outer_diam=200, inner_diam=0, \
    stage_pos=[0,-10,10], radians_per_um=None, NA=0.58, wavelength=0.633, sz=512, \
    fresnel_focal_length=None, um_per_pix=6.0):
    """
    Create a set of simulated phase diversity images. 
    
    Note that dimensions here are in microns.
    
    Parameters
    ----------
    foc_offsets: (n_images-1) numpy array
        Focus offset in radians for the second and subsequent images
    xt_offsets: (n_images-1) numpy array
        X tilt offset
    yt_offsets: (n_images-1) numpy array
        Y tilt offset
    phase_zernikes: numpy array
        Zernike terms for phase, excluding piston.
    amp_zernikes: numpy array
        Zernike terms for amplitude, including overall normalisation.
    outer_rad, inner_rad: float
        Inner and outer radius of annular pupil in pixels. Note that a better
        model would have a (slightly) variable pupil size as the focus changes.
    radians_per_micron: float
        Radians in focus term per micron of stage movement. This is
        approximately 2*np.pi * NA^2 / wavelength.
    stage_pos: (n_images) numpy array
        Nominal stage position in microns.
    fresnel_focal_length: float
        Focal length in microns if we are in the Fresnel regime. If this is None, 
        a Fraunhofer calculation will be made.
    um_per_pix: float
        If we are in the Fresnel regime, we need to define the pixel scale of the 
        input pupil.
    """
    #Firstly, sort out focus, and tilt offsets. This focus offset is a little of a 
    #guess...
    if radians_per_um is None:
        radians_per_um = np.pi*NA**2/wavelength
    total_focus = np.array(stage_pos) * radians_per_um
    total_focus[1:] += np.array(foc_offsets)
    
    #Add a zero (for ref image) to the tilt offsets
    xt = np.concatenate([[0],xt_offsets])
    yt = np.concatenate([[0],yt_offsets])
    
    #Create the amplitude zernike array. Normalise so that the
    #image sum is zero for a evenly illuminated pupil (amplitude zernikes
    #all 0).
    pup_even = circle(sz, outer_diam, interp_edge=True) - \
        circle(sz, inner_diam, interp_edge=True)
    pup_even /= np.sqrt(np.sum(pup_even**2))*sz
    pup = pup_even*zernike_amp(sz, amp_zernikes, diam=outer_diam)
    
    #Needed for the Fresnel calculation
    flux_norm = np.sum(pup**2)/np.sum(pup_even**2)
    
    #Prepare for fresnel propagation if needed.
    if fresnel_focal_length is not None:
        lens = FocusingLens(sz, um_per_pix, um_per_pix, fresnel_focal_length, wavelength)
        print("Using Fresnel propagation...")
    
    #Now iterate through the images at different foci.
    n_ims = len(total_focus)
    ims = np.zeros( (n_ims, sz, sz) )
    for i in range(n_ims):
        #Phase zernikes for this image
        im_phase_zernikes = np.concatenate([[0.], phase_zernikes])
        im_phase_zernikes[1] += xt[i]
        im_phase_zernikes[2] += yt[i]
        im_phase_zernikes[4] += total_focus[i]
        wf = pup*zernike_wf(sz, im_phase_zernikes, diam=outer_diam)
        if fresnel_focal_length is None:
            ims[i] = np.fft.fftshift(np.abs(np.fft.fft2(wf))**2)
        else:
            #For a Fresnel propagation, we need to normalise separately,
            #because the lens class was written with inbuilt normalisation.
            ims[i] = lens.focus(wf) * flux_norm
    return ims

def fourier_wf(sz,xcyc_aperture,ycyc_aperture,amp,phase):
    """This function creates a phase aberration, centered on the
    middle of a python array
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    xcyc_aperture: float
        cycles per aperture in the x direction.
    ycyc_aperture: float
        cycles per aperture in the y direction.
    amp: float
        amplitude of the aberration in radians
    phase: float
        phase of the aberration in radians.
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array circular pupil mask
    """
    x = np.arange(sz) - sz//2
    xy = np.meshgrid(x,x)
    xx = xy[0]
    yy = xy[1]
    zz = 2*np.pi*(xx*xcyc_aperture/sz + yy*ycyc_aperture/sz)
    aberration = np.exp( 1j * amp * (np.cos(phase)*np.cos(zz) + np.sin(phase)*np.sin(zz)))
    return aberration


def gmt(dim,widths=None,pistons=[0,0,0,0,0,0],m_pix=None):
    """This function creates a GMT pupil.
    http://www.gmto.org/Resources/GMT-ID-01467-Chapter_6_Optics.pdf
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        diameter of the primary mirror (scaled to 25.448m)
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array circular pupil mask
    """
    #The geometry is complex... with eliptical segments due to their tilt.
    #We'll just approximate by segments of approximately the right size.
    pupils=[]
    if m_pix:
        widths = 25.448/m_pix
    elif not widths:
        print("ERROR: Must set widths or m_pix")
        raise UserWarning
    try:
        awidth = widths[0]
    except:
        widths = [widths]
    for width in widths:
        segment_dim = width*8.27/25.448
        segment_sep = width*(8.27 + 0.3)/25.448
        obstruct = width*3.2/25.448
        rollx = int(np.round(np.sqrt(3)/2.0*segment_sep))
        one_seg = circle(dim, segment_dim)
        pupil = one_seg - circle(dim, obstruct) + 0j
        pupil += np.exp(1j*pistons[0])*np.roll(np.roll(one_seg,  int(np.round(0.5*segment_sep)), axis=0),rollx, axis=1)
        pupil += np.exp(1j*pistons[1])*np.roll(np.roll(one_seg, -int(np.round(0.5*segment_sep)), axis=0),rollx, axis=1)
        pupil += np.exp(1j*pistons[4])*np.roll(np.roll(one_seg,  int(np.round(0.5*segment_sep)), axis=0),-rollx, axis=1)
        pupil += np.exp(1j*pistons[3])*np.roll(np.roll(one_seg, -int(np.round(0.5*segment_sep)), axis=0),-rollx, axis=1)
        pupil += np.exp(1j*pistons[5])*np.roll(one_seg, int(segment_sep), axis=0)
        pupil += np.exp(1j*pistons[2])*np.roll(one_seg, -int(segment_sep), axis=0)
        pupils.append(pupil)
    return np.array(pupils)

#--- Start masks ---

def mask2s(dim):
    """ Returns 4 pupil mask that split the pupil into halves.
    """
    masks = np.zeros( (4,dim,dim) )
    masks[0,0:dim/2,:]=1
    masks[1,dim/2:,:]=1
    masks[2,:,0:dim/2]=1
    masks[3,:,dim/2:]=1
    return masks

def mask6s(dim):
    """ Returns 4 pupil mask that split the pupil into halves, with a
    six-way symmetry
    """
    masks = np.zeros( (4,dim,dim) )
    x = np.arange(dim) - dim//2
    xy = np.meshgrid(x,x)
    theta = np.arctan2(xy[0],xy[1])
    twelfths = ( (theta + np.pi)/2/np.pi*12).astype(int)
    masks[0,:,:]=(twelfths//2) % 2
    masks[1,:,:]=(twelfths//2 + 1) % 2
    masks[2,:,:]=((twelfths+1)//2) % 2
    masks[3,:,:]=((twelfths+1)//2 + 1) % 2
    
    return masks

def angel_mask(sz,m_per_pix,diam=25.5):
    """Create a mask like Roger Angel et al's original GMT tilt and piston sensor.
    """
    diam_in_pix = diam/m_per_pix
    inner_circ = circle(sz,int(round(diam_in_pix/3)))
    outer_an = circle(sz,int(round(diam_in_pix))) - inner_circ
    mask6s = mask6s(sz)
    masks = np.array([inner_circ + outer_an*mask6s[2,:,:],inner_circ + outer_an*mask6s[3,:,:],outer_an])
    return masks

def angel_mask_mod(sz,wave,diam=25.5):
    """Create a mask like Roger Angel et al's original GMT tilt and piston sensor, except 
    we 50/50 split the inner segment.
    """
    diam_in_pix = diam/m_per_pix
    inner_circ = circle(sz,int(round(diam_in_pix/3)))
    outer_an = circle(sz,int(round(diam_in_pix))) - inner_circ
    mask6s = mask6s(sz)
    masks = np.array([0.5*inner_circ + outer_an*mask6s[2,:,:],0.5*inner_circ + outer_an*mask6s[3,:,:]])
    return masks

def diversity_mask(sz,m_per_pix,defocus=2.0):
    """Create a traditional phase diversity mask.
    """
    wf1 = curved_wf(sz,m_per_pix,defocus=defocus)
    wf2 = curved_wf(sz,m_per_pix,defocus=-defocus)
    masks = np.array([wf1,wf2])
    return masks
    
#--- End Masks ---

def km1d(sz, r_0_pix=None):
    """
    Algorithm:
        y(midpoint) = ( y(x1) + y(x2) )/2 + 0.4542*Z, where 
            0.4542 = sqrt( 1 - 2^(5/3) / 2 )
    """
    if sz != 2**int(np.log2(sz)):
        raise UserWarning("Size must be within a factor of 2")
    #Temporary code.
    wf = kmf(sz, r_0_pix=r_0_pix)
    return wf[0]

def kmf(sz, L_0=np.inf, r_0_pix=None):
    """This function creates a periodic wavefront produced by Kolmogorov turbulence. 
    It SHOULD normalised so that the variance at a distance of 1 pixel is 1 radian^2.
    To scale this to an r_0 of r_0_pix, multiply by sqrt(6.88*r_0_pix**(-5/3))
    
    The value of 1/15.81 in the code is (I think) a numerical approximation for the 
    value in e.g. Conan00 of np.sqrt(0.0229/2/np.pi)
    
    Parameters
    ----------
    sz: int
        Size of the 2D array
        
    l_0: (optional) float
        The von-Karmann outer scale. If not set, the structure function behaves with
        an outer scale of approximately half (CHECK THIS!) pixels. 
   
    r_0_pix: (optional) float
	The Fried r_0 parameter in units of pixels.
 
    Returns
    -------
    wavefront: float array (sz,sz)
        2D array wavefront, in units of radians. i.e. a complex electric field based
        on this wavefront is np.exp(1j*kmf(sz))
    """
    xy = np.meshgrid(np.arange(sz/2 + 1)/float(sz), (((np.arange(sz) + sz/2) % sz)-sz/2)/float(sz))
    dist2 = np.maximum( xy[1]**2 + xy[0]**2, 1e-12)
    ft_wf = np.exp(2j * np.pi * np.random.random((sz,sz//2+1)))*dist2**(-11.0/12.0)*sz/15.81
    ft_wf[0,0]=0
    if r_0_pix is None:
        return np.fft.irfft2(ft_wf)
    else:
        return np.fft.irfft2(ft_wf) * np.sqrt(6.88*r_0_pix**(-5/3.))

def von_karman_structure(B, r_0=1.0, L_0=1e6):
    """The Von Karan structure function, from Conan et al 2000"""
    return 0.1717*(r_0/L_0)**(-5/3.)*(1.005635 - (2*np.pi*B/L_0)**(5/6.)*special.kv(5/6.,2*np.pi*B/L_0))
 
def test_kmf(sz,ntests):
    """Test the kmf. The variance at sz/4 is down by a factor of 0.35 over the 
    Kolmogorov function."""
    vars_1pix = np.zeros(ntests)
    vars_quarter = np.zeros(ntests)
    for i in range(ntests):
        wf = kmf(sz)
        vars_1pix[i] = 0.5*(np.mean((wf[1:,:] - wf[:-1,:])**2) + \
                            np.mean((wf[:,1:] - wf[:,:-1])**2))
        vars_quarter[i] = 0.5*(np.mean((np.roll(wf,sz//4,axis=0) - wf)**2) + \
                            np.mean((np.roll(wf,sz//4,axis=1) - wf)**2))
                        
    print("Mean var: {0:7.3e} Sdev var: {1:7.3e}".format(np.mean(vars_1pix),np.std(vars_1pix)))
    print("Variance at sz//4 decreased by: {0:7.3f}".\
        format(np.mean(vars_quarter)/np.mean(vars_1pix)/(sz/4)**(5./3.)))
        
def moffat(theta, hw, beta=4.0):
    """This creates a moffatt function for simulating seeing.
    The output is an array with the same dimensions as theta.
    Total Flux" is set to 1 - this only applies if sampling
    of thetat is 1 per unit area (e.g. arange(100)).
    
    From Racine (1996), beta=4 is a good approximation for seeing
    
    Parameters
    ----------
    theta: float or float array
        Angle at which to calculate the moffat profile (same units as hw)
    hw: float
        Half-width of the profile
    beta: float
        beta parameters
    
    """
    denom = (1 + (2**(1.0/beta) - 1)*(theta/hw)**2)**beta
    return (2.0**(1.0/beta)-1)*(beta-1)/np.pi/hw**2/denom
    
def moffat2d(sz,hw, beta=4.0):
    """A 2D version of a moffat function
    """
    x = np.arange(sz) - sz/2.0
    xy = np.meshgrid(x,x)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    return moffat(r, hw, beta=beta)
       
def snell(u, f, n_i, n_f):
    """Snell's law at an interface between two dielectrics
    
    Parameters
    ----------
    u: float array(3)
        Input unit vector
    f: float array(3)
        surface normal  unit vector
    n_i: float
        initial refractive index
    n_f: float
        final refractive index.
    """
    u_p = u - np.sum(u*f)*f
    u_p /= np.sqrt(np.sum(u_p**2))
    theta_i = np.arccos(np.sum(u*f))
    theta_f = np.arcsin(n_i*np.sin(theta_i)/n_f)
    v = u_p*np.sin(theta_f) + f*np.cos(theta_f)
    return v

def grating_sim(u, l, s, ml_d, refract=False):
    """This function computes an output unit vector based on an input unit
    vector and grating properties.

    Math: v \cdot l = u \cdot l (reflection)
          v \cdot s = u \cdot s + ml_d
    The blaze wavelength is when m \lambda = 2 d sin(theta)
     i.e. ml_d = 2 sin(theta)

    x : to the right
    y : out of page
    z : down the page
    
    Parameters
    ----------
    u: float array(3)
        initial unit vector
    l: float array(3)
        unit vector along grating lines
    s: float array(3)
        unit vector along grating surface, perpendicular to lines
    ml_d: float
        order * \lambda/d
    refract: bool
        Is the grating a refractive grating? 
    """
    if (np.abs(np.sum(l*s)) > 1e-3):    
        print('Error: input l and s must be orthogonal!')
        raise UserWarning
    n = np.cross(s,l)
    if refract:
        n *= -1
    v_l = np.sum(u*l)
    v_s = np.sum(u*s) + ml_d
    v_n = np.sqrt(1-v_l**2 - v_s**2)
    v = v_l*l + v_s*s + v_n*n
    
    return v

def join_bessel(U,V,j):
    """In order to solve the Laplace equation in cylindrical co-ordinates, both the
    electric field and its derivative must be continuous at the edge of the fiber...
    i.e. the Bessel J and Bessel K have to be joined together. 
    
    The solution of this equation is the n_eff value that satisfies this continuity
    relationship"""
    W = np.sqrt(V**2 - U**2)
    return U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
    
def neff(V, accurate_roots=True):
    """For a cylindrical fiber, find the effective indices of all modes for a given value 
    of the fiber V number. 
    
    Parameters
    ----------
    V: float
        The fiber V-number.
    accurate_roots: bool (optional)
        Do we find accurate roots using Newton-Rhapson iteration, or do we just use a 
        first-order linear approach to zero-point crossing?"""
    delu = 0.04
    numu = int(V/delu)
    U = np.linspace(delu/2,V - 1e-6,numu)
    W = np.sqrt(V**2 - U**2)
    all_roots=np.array([])
    n_per_j=np.array([],dtype=int)
    n_modes=0
    for j in range(int(V+1)):
        f = U*special.jn(j+1,U)*special.kn(j,W) - W*special.kn(j+1,W)*special.jn(j,U)
        crossings = np.where(f[0:-1]*f[1:] < 0)[0]
        roots = U[crossings] - f[crossings]*( U[crossings+1] - U[crossings] )/( f[crossings+1] - f[crossings] )
        if accurate_roots:
            for i in range(len(crossings)):
                roots[i] = optimize.brenth(join_bessel, U[crossings[i]], U[crossings[i]+1], args=(V,j))
        
#roots[i] = optimize.newton(join_bessel, root, args=(V,j))
#                except:
#                    print("Problem finding root, trying 1 last time...")
#                    roots[i] = optimize.newton(join_bessel, root + delu/2, args=(V,j))
        #import pdb; pdb.set_trace()
        if (j == 0): 
            n_modes = n_modes + len(roots)
            n_per_j = np.append(n_per_j, len(roots))
        else:
            n_modes = n_modes + 2*len(roots)
            n_per_j = np.append(n_per_j, len(roots)) #could be 2*length(roots) to account for sin and cos.
        all_roots = np.append(all_roots,roots)
    return all_roots, n_per_j
 
def mode_2d(V, r, j=0, n=0, sampling=0.3,  sz=1024):
    """Create a 2D mode profile. 
    
    Parameters
    ----------
    V: Fiber V number
    
    r: core radius in microns
    
    sampling: microns per pixel
    
    n: radial order of the mode (0 is fundumental)
    
    j: azimuthal order of the mode (0 is pure radial modes)
    TODO: Nonradial modes."""
    #First, find the neff values...
    u_all,n_per_j = neff(V)
    
    #Unsigned 
    unsigned_j = np.abs(j)
    th_offset = (j<0) * np.pi/2
    
    #Error check the input.
    if n >= n_per_j[unsigned_j]:
        print("ERROR: this mode is not bound!")
        raise UserWarning
    
    # Convert from float to be able to index
    sz = int(sz)
    
    ix = np.sum(n_per_j[0:unsigned_j]) + n
    U0 = u_all[ix]
    W0 = np.sqrt(V**2 - U0**2)
    x = (np.arange(sz)-sz/2)*sampling/r
    xy = np.meshgrid(x,x)
    r = np.sqrt(xy[0]**2 + xy[1]**2)
    th = np.arctan2(xy[0],xy[1]) + th_offset
    win = np.where(r < 1)
    wout = np.where(r >= 1)
    the_mode = np.zeros( (sz,sz) )
    the_mode[win] = special.jn(unsigned_j,r[win]*U0)
    scale = special.jn(unsigned_j,U0)/special.kn(unsigned_j,W0)
    the_mode[wout] = scale * special.kn(unsigned_j,r[wout]*W0)
    return the_mode/np.sqrt(np.sum(the_mode**2))*np.exp(1j*unsigned_j*th)

def compute_v_number(wavelength_in_mm, core_radius, numerical_aperture):
    """Computes the V number (can be interpreted as a kind of normalized optical frequency) for an optical fibre
    
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength of light in mm
    core_radius: float
        The core radius of the fibre in mm
    numerical_aperture: float
        The numerical aperture of the optical fibre, defined be refractive indices of the core and cladding
        
    Returns
    -------
    v: float
        The v number of the fibre
        
    """
    v = 2 * np.pi / wavelength_in_mm * core_radius * numerical_aperture
    return v
    
def shift_and_ft(im):
    """Sub-pixel shift an image to the origin and Fourier-transform it

    Parameters
    ----------
    im: (ny,nx) float array
    ftpix: optional ( (nphi) array, (nphi) array) of Fourier sampling points. 
    If included, the mean square Fourier phase will be minimised.

    Returns
    ----------
    ftim: (ny,nx/2+1)  complex array
    """
    ny = im.shape[0]
    nx = im.shape[1]
    im = regrid_fft(im,(3*ny,3*nx))
    shifts = np.unravel_index(im.argmax(), im.shape)
    im = np.roll(np.roll(im,-shifts[0]+1,axis=0),-shifts[1]+1,axis=1)
    im = rebin(im,(ny,nx))
    ftim = np.fft.rfft2(im)
    return ftim

def rebin(a, shape):
    """Re-bins an image to a new (smaller) image with summing	

    Originally from:
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array

    Parameters
    ----------
    a: array
        Input image
    shape: (xshape,yshape)
        New shape
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def correct_tip_tilt(turbulent_wf, pupil, size):
    """Given a turbulent wavefront, calculate the tip/tilt (horizontal and vertical slope)
    
    TODO: Only compute turbulence over square immediately surrounding the pupil to save on unnecessary computation
    
    Parameters
    ----------
    turbulent_wf: np.array([[...]...])
        2D square of numbers representing a turbulent patch of atmosphere
    pupil: np.array([[...]...])
        The pupil of the telescope receiving the light pasing through the turbulence.
    size: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
        
    Return
    ------
    corrected_wf: np.array([[...]...])
        Tip/Tilt corrected turbulent_wf
    """
    x = np.arange(size) - size/2
    xy = np.meshgrid(x, x)
    xtilt_func = xy[0]*pupil
    ytilt_func = xy[1]*pupil
    
    xtilt = np.sum(xtilt_func * turbulent_wf)/np.sum(xtilt_func**2)
    ytilt = np.sum(ytilt_func * turbulent_wf)/np.sum(ytilt_func**2)
    
    corrected_wf = turbulent_wf*pupil - ytilt_func*ytilt - xtilt_func*xtilt
    
    return corrected_wf   

def apply_and_scale_turbulent_ef(turbulence, npix, wavelength, dx, seeing):
    """ Applies an atmosphere in the form of Kolmogorov turbulence to an initial wavefront and scales
    
    Parameters
    ----------
    npix: integer
        The size of the square of Kolmogorov turbulence generated
    wavelength: float
        The wavelength in mm. Amount of atmospheric distortion depends on the wavelength. 
    dx: float
        Resolution in mm/pixel
    seeing: float
        Seeing in arcseconds before magnification
    Returns
    -------
    turbulent_ef or 1.0: np.array([[...]...]) or 1.0
        Return an array of phase shifts in imperfect seeing, otherwise return 1.0, indicating no change to the incident wave.
    """
    if seeing > 0.0:
        # Convert seeing to radians
        seeing_in_radians = np.radians(seeing/3600.)
        
        # Generate the Kolmogorov turbulence
        #turbulence = optics_tools.kmf(npix)
        
        # Calculate r0 (Fried's parameter), which is a measure of the strength of seeing distortions
        r0 = 0.98 * wavelength / seeing_in_radians 
        
        # Apply the atmosphere and scale
        wf_in_radians = turbulence * np.sqrt(6.88*(dx/r0)**(5.0/3.0))
            
        # Convert the wavefront to an electric field
        turbulent_ef = np.exp(1.0j * wf_in_radians)
        
        return turbulent_ef
    else:
        # Do not apply phase distortions --> multiply by unity
        return 1.0

def calculate_fibre_mode(wavelength_in_mm, fibre_core_radius, numerical_aperture, npix, dx):
    """Computes the mode of the optical fibre.
    Parameters
    ----------
    wavelength_in_mm: float
        The wavelength in mm
    fibre_core_radius: float
        The radius of the fibre core in mm
    numerical_aperture: float
        The numerical aperture of the fibre
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel    
    Returns
    -------
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre
    """
    # Calculate the V number for the model
    v = compute_v_number(wavelength_in_mm, fibre_core_radius, numerical_aperture)
    
    # Use the V number to calculate the mode
    fibre_mode = mode_2d(v, fibre_core_radius, sampling=dx, sz=npix)

    return fibre_mode


def compute_coupling(npix, dx, electric_field, lens_width, fibre_mode, x_offset, y_offset):
    """Computes the coupling between the electric field and the optical fibre using an overlap integral.
    
    Parameters
    ----------
    npix: int
        Size of input_wf per side, preferentially a power of two (npix=2**n)
    dx: float
        Resolution of the wave in mm/pixel      
    electric_field: np.array([[...]...])
        The electric field at the fibre plane
    lens_width: float
        The width of the a single microlens (used for minimising the unnecessary calculations)
    fibre_mode: np.array([[...]...])
        The mode of the optical fibre   
    x_offset: int
        x offset of the focal point at the fibre plane relative to the centre of the microlens.
    y_offset: int
        y offset of the focal point at the fibre plane relative to the centre of the microlens.           
   
    Returns
    -------
    coupling: float
        The coupling between the fibre mode and the electric_field (Max 1)
    """
    npix = int(npix)
    
    # Crop the electric field to the central 1/4
    low = npix//2 - int(lens_width / dx / 2) #* 3/8
    upper = npix//2 + int(lens_width / dx / 2) #* 5/8
    
    # Compute the fibre mode and shift (if required)
    fibre_mode = fibre_mode[(low + x_offset):(upper + x_offset), (low + y_offset):(upper + y_offset)]
    
    # Compute overlap integral - denominator first
    den = np.sum(np.abs(fibre_mode)**2) * np.sum(np.abs(electric_field)**2)
    
    #Crop the electric field and compute the numerator
    #electric_field = electric_field[low:upper,low:upper]
    num = np.abs(np.sum(fibre_mode*np.conj(electric_field)))**2

    coupling = num / den
    
    return coupling

def nglass(l, glass='sio2'):
    """Refractive index of fused silica and other glasses. Note that C is
    in microns^{-2}
    
    Parameters
    ----------
    l: wavelength 
    """
    try:
        nl = len(l)
    except:
        l = [l]
        nl=1
    l = np.array(l)
    if (glass == 'sio2'):
        B = np.array([0.696166300, 0.407942600, 0.897479400])
        C = np.array([4.67914826e-3,1.35120631e-2,97.9340025])
    elif (glass == 'bk7'):
        B = np.array([1.03961212,0.231792344,1.01046945])
        C = np.array([6.00069867e-3,2.00179144e-2,1.03560653e2])
    elif (glass == 'nf2'):
        B = np.array( [1.39757037,1.59201403e-1,1.26865430])
        C = np.array( [9.95906143e-3,5.46931752e-2,1.19248346e2])
    elif (glass == 'nsf11'):
        B = np.array([1.73759695E+00,   3.13747346E-01, 1.89878101E+00])
        C = np.array([1.31887070E-02,   6.23068142E-02, 1.55236290E+02])
    elif (glass == 'ncaf2'):
        B = np.array([0.5675888, 0.4710914, 3.8484723])
        C = np.array([0.050263605,  0.1003909,  34.649040])**2
    elif (glass == 'mgf2'):
        B = np.array([0.48755108,0.39875031,2.3120353])
        C = np.array([0.04338408,0.09461442,23.793604])**2
    elif (glass == 'npk52a'):
        B = np.array([1.02960700E+00,1.88050600E-01,7.36488165E-01])
        C = np.array([5.16800155E-03,1.66658798E-02,1.38964129E+02])
    elif (glass == 'psf67'):
        B = np.array([1.97464225E+00,4.67095921E-01,2.43154209E+00])
        C = np.array([1.45772324E-02,6.69790359E-02,1.57444895E+02])
    elif (glass == 'npk51'):
        B = np.array([1.15610775E+00,1.53229344E-01,7.85618966E-01])
        C = np.array([5.85597402E-03,1.94072416E-02,1.40537046E+02])
    elif (glass == 'nfk51a'):
        B = np.array([9.71247817E-01,2.16901417E-01,9.04651666E-01])
        C = np.array([4.72301995E-03,1.53575612E-02,1.68681330E+02])
    elif (glass == 'si'): #https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg
        B = np.array([10.6684293,0.0030434748,1.54133408])
        C = np.array([0.301516485,1.13475115,1104])**2
    #elif (glass == 'zns'): #https://refractiveindex.info/?shelf=main&book=ZnS&page=Debenham
    #    B = np.array([7.393, 0.14383, 4430.99])
    #    C = np.array([0, 0.2421, 36.71])**2
    elif (glass == 'znse'): #https://refractiveindex.info/?shelf=main&book=ZnSe&page=Connolly
        B = np.array([4.45813734,0.467216334,2.89566290])
        C = np.array([0.200859853,0.391371166,47.1362108])**2
    elif (glass == 'noa61'):
        n = 1.5375 + 8290.45/(l*1000)**2 - 2.11046/(l*1000)**4
        return n
    else:
        print("ERROR: Unknown glass {0:s}".format(glass))
        raise UserWarning
    n = np.ones(nl)
    for i in range(len(B)):
            n += B[i]*l**2/(l**2 - C[i])
    return np.sqrt(n)

#The following is directly from refractiveindex.info, and copied here because of
#UTF-8 encoding that doesn't seem to work with my python 2.7 installation.
#Author: Mikhail Polyanskiy
#(Ciddor 1996, https://doi.org/10.1364/AO.35.001566)

def Z(T,p,xw): #compressibility
    t=T-273.15
    a0 = 1.58123e-6   #K.Pa^-1
    a1 = -2.9331e-8   #Pa^-1
    a2 = 1.1043e-10   #K^-1.Pa^-1
    b0 = 5.707e-6     #K.Pa^-1
    b1 = -2.051e-8    #Pa^-1
    c0 = 1.9898e-4    #K.Pa^-1
    c1 = -2.376e-6    #Pa^-1
    d  = 1.83e-11     #K^2.Pa^-2
    e  = -0.765e-8    #K^2.Pa^-2
    return 1-(p/T)*(a0+a1*t+a2*t**2+(b0+b1*t)*xw+(c0+c1*t)*xw**2) + (p/T)**2*(d+e*xw**2)


def nm1_air(wave,t,p,h,xc):
    # wave: wavelength, 0.3 to 1.69 mu m 
    # t: temperature, -40 to +100 deg C
    # p: pressure, 80000 to 120000 Pa
    # h: fractional humidity, 0 to 1
    # xc: CO2 concentration, 0 to 2000 ppm

    sigma = 1/wave           #mu m^-1
    
    T= t + 273.15     #Temperature deg C -> K
    
    R = 8.314510      #gas constant, J/(mol.K)
    
    k0 = 238.0185     #mu m^-2
    k1 = 5792105      #mu m^-2
    k2 = 57.362       #mu m^-2
    k3 = 167917       #mu m^-2
 
    w0 = 295.235      #mu m^-2
    w1 = 2.6422       #mu m^-2
    w2 = -0.032380    #mu m^-4
    w3 = 0.004028     #mu m^-6
    
    A = 1.2378847e-5  #K^-2
    B = -1.9121316e-2 #K^-1
    C = 33.93711047
    D = -6.3431645e3  #K
    
    alpha = 1.00062
    beta = 3.14e-8       #Pa^-1,
    gamma = 5.6e-7        #deg C^-2

    #saturation vapor pressure of water vapor in air at temperature T
    if(t>=0):
        svp = np.exp(A*T**2 + B*T + C + D/T) #Pa
    else:
        svp = 10**(-2663.5/T+12.537)
    
    #enhancement factor of water vapor in air
    f = alpha + beta*p + gamma*t**2
    
    #molar fraction of water vapor in moist air
    xw = f*h*svp/p
    
    #refractive index of standard air at 15 deg C, 101325 Pa, 0% humidity, 450 ppm CO2
    nas = 1 + (k1/(k0-sigma**2)+k3/(k2-sigma**2))*1e-8
    
    #refractive index of standard air at 15 deg C, 101325 Pa, 0% humidity, xc ppm CO2
    naxs = 1 + (nas-1) * (1+0.534e-6*(xc-450))
    
    #refractive index of water vapor at standard conditions (20 deg C, 1333 Pa)
    nws = 1 + 1.022*(w0+w1*sigma**2+w2*sigma**4+w3*sigma**6)*1e-8
    
    Ma = 1e-3*(28.9635 + 12.011e-6*(xc-400)) #molar mass of dry air, kg/mol
    Mw = 0.018015                            #molar mass of water vapor, kg/mol
    
    Za = Z(288.15, 101325, 0)                #compressibility of dry air
    Zw = Z(293.15, 1333, 1)                  #compressibility of pure water vapor
    
    #Eq.4 with (T,P,xw) = (288.15, 101325, 0)
    rhoaxs = 101325*Ma/(Za*R*288.15)           #density of standard air
    
    #Eq 4 with (T,P,xw) = (293.15, 1333, 1)
    rhows  = 1333*Mw/(Zw*R*293.15)             #density of standard water vapor
    
    # two parts of Eq.4: rho=rhoa+rhow
    rhoa   = p*Ma/(Z(T,p,xw)*R*T)*(1-xw)       #density of the dry component of the moist air    
    rhow   = p*Mw/(Z(T,p,xw)*R*T)*xw           #density of the water vapor component
    
    nprop = (rhoa/rhoaxs)*(naxs-1) + (rhow/rhows)*(nws-1)
    
    return nprop

class FresnelPropagator(object):
    """Propagate a wave by Fresnel diffraction"""
    def __init__(self,sz,m_per_pix, d, wave,nthreads=nthreads):
        """Initiate this fresnel_propagator for a particular wavelength, 
        distance etc.
    
        Parameters
        ----------
        wf: float array
            
        m_per_pix: float
            Scale of the pixels in the input wavefront in metres.
        d: float
            Distance to propagate the wavefront.
        wave: float
            Wavelength in metres.
        nthreads: int
            Number of threads. 
        """
        self.sz = sz
        self.nthreads=nthreads
        #Co-ordinate axis of the wavefront Fourier transform. Not that 0 must be in the corner.
        #x is in cycles per wavefront dimension.
        x = (((np.arange(sz)+sz/2) % sz) - sz/2)/m_per_pix/sz
        xy = np.meshgrid(x,x)
        uu =np.sqrt(xy[0]**2 + xy[1]**2)
        self.h_ft = np.exp(1j*np.pi*uu**2*wave*d)
    
    def propagate(self,wf):
        """Propagate a wavefront, according to the parameters established on the
        __init__. No error checking for speed.
        
        Parameters
        ----------
        wf: complex array
            Wavefront, i.e. a complex electric field in the scalar approximation.
        
        Returns
        -------
        wf_new: float array
            Wavefront after propagating.
        """

        if (wf.shape[0] != self.sz | wf.shape[1] != self.sz):
            print("ERROR: Input wavefront must match the size!")
            raise UserWarning
        if (self.nthreads>0):
            g_ft = pyfftw.interfaces.numpy_fft.fft2(wf,threads=self.nthreads)*self.h_ft
            wf_new = pyfftw.interfaces.numpy_fft.ifft2(g_ft,threads=self.nthreads)
        else:
            g_ft = np.fft.fft2(wf)*self.h_ft
            wf_new = np.fft.ifft2(g_ft)
        return wf_new

class FocusingLens(FresnelPropagator):
    def __init__(self,sz,m_per_pix_pup, m_per_pix_im, f, wave,nthreads=nthreads):
        """Use Fresnel Diffraction to come to focus. 
    
        We do this by creating a new lens of focal length mag * f, where mag is the 
        magnification between pupil and image plane.
        """
        f_new=f * m_per_pix_pup/m_per_pix_im
        
        #Initialise the parent class.
        super(FocusingLens, self).__init__(sz, m_per_pix_pup, f_new, wave,nthreads=nthreads)
        #super(FresnelPropagator, self).__init__(m_per_pix_pup, f_new, wave,nthreads=nthreads)
        #FresnelPropagator.__init__(self, sz,m_per_pix_pup, f_new, wave,nthreads=nthreads)

        #Create our curved wavefront.
        self.lens = curved_wf(sz, m_per_pix_pup, f_length=f_new, wave=wave)
        self.sz=sz
        
    def focus(self, wf):
        """Return a normalised image"""
        if (wf.shape[0] != self.sz) or (wf.shape[1] != self.sz):
            raise UserWarning("Incorrect Wavefront Shape!")
        im = np.abs(self.propagate(wf*self.lens))**2
        return im/np.sum(im)

def focusing_propagator(sz, m_per_pix_pup, m_per_pix_im, f, wave):
    """Create a propagator that propagates to focus, adjusting
    the focal length for the new image scale using the thin lens formula.
    
    The new lens has a focal length of mag * f, where mag is the magnification.
    
    FIXME: Remove this if FocusingLens works.
    
    Returns
    -------
    lens: (sz,sz) numpy complex array
        Multiply the final pupil by this prior to applying the propagator
        
    to_focus: FresnelPropagator
        use np.abs(to_focus.propagate(wf*lens))**2 to create the image
    """
    #Create a new focal length that is longer according to the magnification.
    f_new = f * m_per_pix_pup/m_per_pix_im
    
    #Create the curved wavefront.
    lens = curved_wf(sz, m_per_pix_pup, f_length=f_new, wave=wave)
    
    #Create the propagator
    to_focus = FresnelPropagator(sz,m_per_pix, f_new, wave)
    
    return lens, to_focus
    
class Base(object):
    def __init__(self):
        print("Base created")

class ChildA(Base):
    def __init__(self):
        Base.__init__(self)

class ChildB(Base):
    def __init__(self):
        super(ChildB, self).__init__()
        
def fresnel_reflection(n1, n2, theta=0):
    """
    Parameters
    ----------
    theta: float
        incidence angle in degrees
    
    Returns 
    -------
    Rp: float
        s (perpendicular) plane reflection
    Rs: float
        p (parallel) plane reflection
    """
    th = np.radians(theta)
    sqrt_term = np.sqrt(1-(n1/n2*np.sin(th))**2)
    Rs = (n1*np.cos(th) - n2*sqrt_term)**2/(n1*np.cos(th) + n2*sqrt_term)**2
    Rp = (n1*sqrt_term - n2*np.cos(th))**2/(n1*sqrt_term + n2*np.cos(th))**2
    return Rs, Rp
