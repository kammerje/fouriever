"""Useful methods to assist with various functions.

Author: 
Dr Michael Ireland
Adam Rains 
"""
from __future__ import division, print_function
import pylab as pl
#import cv2
import glob
import numpy as np
import pdb
import matplotlib.pyplot as plt
H = 6.626e-34 #Planck constant in SI
C = 3e8 #Speed of light in m/s
K_B = 1.38e-23 #Boltzmann constant in SI

def from_arcsec(arcsec):
    return np.radians(arcsec/60.0/60.0)
    
def to_arcsec(radians):
    return np.degrees(radians)*60.0*60.0

def bb_photonrate(T, wave=None, nu=None, delta_wave=None, delta_nu=1.0):
    """Find the photon rate in photons per spatial and temporal
    bandwidth
    """
    if wave is not None:
        nu = C/wave
    if delta_wave is not None:
        wave = C/nu
        delta_nu = nu*delta_wave/wave
    return delta_nu*2/(np.exp(H*nu/K_B/T)-1)

def save_plot(data, title, directory, imagename, format='.jpg'):
    """
    
    Parameters
    ----------
    
    data: 2D numpy.array
        The data to be plotted and saved. Will not accept complex numbers.
    title: string
        The title of the plot.
    directory: string
        The directory to save the plot to, ending with a "/"
    imagename: string
        The name of the plot to be saved.
    format: string  
        The file formate of the saved image.
    """
    
    pl.clf()
    pl.imshow(data) 
    pl.title(title)
    pl.savefig( (directory + (imagename) + format ) )

def create_movie(directory, image_format='jpg', fps=5, video_name='video.avi'):
    """Create a movie using a sequence of images

    Parameters
    ----------
    directory: string
        The directory to load the images from and save the video to.
    image_format: string
        The file format of the image files.
    fps: integer
        Frames Per Second of the created video
    video_name: string  
        The filename of the resulting video (including format)
    """
    
    # Get a list of the file paths of all images matching the file format and sort them
    image_paths = glob.glob( (directory + "*." +image_format) )
    image_paths.sort()
    
    # Create the video writer
    # cv2.VideoWriter(filename, fourcc, fps, frame_size, is_color)
    height, width, layers = cv2.imread(image_paths[0]).shape
    video = cv2.VideoWriter( (directory + video_name), -1, fps, (width, height), 1)
    
    # Read each image and write to the video
    for i in image_paths:
        image = cv2.imread(i)
        
        video.write(image)
    
    # Clean up
    cv2.destroyAllWindows()
    video.release()
    
def circle(dim,width,interp_edge=False):
    """This function creates a circle.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        diameter of the circle
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array circular pupil mask
    """
    x = np.arange(dim)-dim//2
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    if interp_edge:
        circle = np.sqrt((width/2.0)**2) - np.sqrt(xx**2+yy**2) + 0.5
        circle = np.maximum(np.minimum(circle, 1),0)
    else:
        circle = ((xx**2+yy**2) < (width/2.0)**2).astype(float)
    return circle
    
def square(dim, width):
    """This function creates a square.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        width of the square
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array square pupil mask
    """
    x = np.arange(dim)-dim//2
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    w = np.where( (yy < width/2) * (yy > (-width/2)) * (xx < width/2) * (xx > (-width/2)))
    square = np.zeros((dim,dim))
    square[w] = 1.0
    return square
    
def gauss(dim,width):
    """This creates a Gausssian beam. Width is the 1/e^2 intensity
    diameter. The electric field is returned. 
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        width of the square
    
    Returns
    -------
    pupil: float array (sz,sz)
        2D array square pupil mask
    """
    x = np.arange(dim) - dim//2
    xy = np.meshgrid(x,x)
    rr = np.sqrt(xy[0]**2 + xy[1]**2)
    beam = np.exp(-(rr/width)**2)
    return beam
    
def hexagon(dim, width, interp_edge=True):
    """This function creates a hexagon.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        flat-to-flat width of the hexagon
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array hexagonal pupil mask
    """
    x = np.arange(dim)-dim/2.0
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    hex = np.zeros((dim,dim))
    scale=1.5
    offset = 0.5
    if interp_edge:
        #!!! Not fully implemented yet. Need to compute the orthogonal distance 
        #from each line and accurately find fractional area of each pixel.
        hex = np.minimum(np.maximum(width/2 - yy + offset,0),1) * \
            np.minimum(np.maximum(width/2 + yy + offset,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width-np.sqrt(3)*xx + yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx - yy + offset)*scale,0),1) * \
            np.minimum(np.maximum((width+np.sqrt(3)*xx + yy + offset)*scale,0),1)
    else:
        w = np.where( (yy < width/2) * (yy > (-width/2)) * \
         (yy < (width-np.sqrt(3)*xx)) * (yy > (-width+np.sqrt(3)*xx)) * \
         (yy < (width+np.sqrt(3)*xx)) * (yy > (-width-np.sqrt(3)*xx)))
        hex[w]=1.0
    return hex

def octagon(dim, width):
    """This function creates a hexagon.
    
    Parameters
    ----------
    dim: int
        Size of the 2D array
    width: int
        flat-to-flat width of the hexagon
        
    Returns
    -------
    pupil: float array (sz,sz)
        2D array hexagonal pupil mask
    """
    x = np.arange(dim)-dim/2.0
    xy = np.meshgrid(x,x)
    xx = xy[1]
    yy = xy[0]
    w = np.where((yy < width/2) * (yy > (-width/2)) * \
                 (xx < width/2) * (xx > (-width/2)) * \
                 (yy < width/np.sqrt(2) - xx) * \
                 (yy > -width/np.sqrt(2) + xx) * \
                 (yy > -width/np.sqrt(2) - xx) * \
                 (yy < width/np.sqrt(2) + yy))[0]
    
    oct = (yy < width/2) * (yy > (-width/2)) * \
          (xx < width/2) * (xx > (-width/2)) * \
          (yy < width/np.sqrt(2) - xx) * \
          (yy > -width/np.sqrt(2) + xx) * \
          (yy > -width/np.sqrt(2) - xx) * \
          (yy < width/np.sqrt(2) + xx)
    oct = oct.astype(float)
    return oct


def annulus(npix, r_large, r_small):
    """Compute the input electric field (annulus x distorted wavefront)
    """ 
    annulus = circle(npix, r_large) - circle(npix, r_small) 

    return annulus    
    
def rotate_xz(u, theta_deg):
    """Rotates a vector u in the x-z plane, clockwise where x is up and
    z is right"""
    th = np.radians(theta_deg)
    M = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    return np.dot(M, u)

def regrid_fft(im,new_shape):
    """Regrid onto a larger number of pixels using an fft. This is optimal
    for Nyquist sampled data.

    Parameters
    ----------
    im: array
        The input image.
    new_shape: (new_y,new_x)
        The new shape

    Notes
    ------
    TODO: This should work with an arbitrary number of dimensions
    """
    ftim = np.fft.rfft2(im)
    new_ftim = np.zeros((new_shape[0], new_shape[1]//2 + 1),dtype='complex')
    new_ftim[0:ftim.shape[0]//2, 0:ftim.shape[1]] = \
        ftim[0:ftim.shape[0]//2, 0:ftim.shape[1]]
    new_ftim[new_shape[0]-ftim.shape[0]//2:, 0:ftim.shape[1]] = \
        ftim[ftim.shape[0]//2:, 0:ftim.shape[1]]
    new_im = np.fft.irfft2(new_ftim)
    return new_im*( np.prod(new_shape)/np.prod(im.shape) )
    
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
    
def interpolate_by_2x(array_before, npix):
    """Expands a complex array by a factor of 2, using a Fourier Transform to interpolate
    in-between pixels.
    
    Parameters
    ----------
    array_before: 2D numpy.ndarray
        The electric field before cropping and interpolation
   
   Returns
    -------
    efield_after: 2D numpy.ndarray
        The electric field after cropping and interpolation          
    """
    # Split into real and imaginary components
    array_real = array_before.real
    array_imag = array_before.imag
    
    new_npix = npix * 2
    
    # Regrid the real and imaginary components separately, expanding the crop back to the original field size (doubling)
    new_array_real = regrid_fft(array_real, [new_npix, new_npix])
    new_array_imag = regrid_fft(array_imag, [new_npix, new_npix])
    
    # Recombine real and imaginary components and return result. Multiply by 2
    # because of the way that regrid_fft rescales.
    array_after = (new_array_real + 1j * new_array_imag) * 2.0
    
    return array_after    