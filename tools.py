#!/usr/bin/env python3

''' Tools to be used in align.py and cc2d.py '''

import warnings

import numpy as np
try:
    from astropy.io import fits
except ModuleNotFoundError:
    warnings.warn('could not import astropy.io.fits')

def save_fits_cube(data, filename, overwrite=False):
    ''' Save a cube to a FITS.

    Parameters
    ==========
    data : ndarray
        The data to save
    filename : str
        The destination where to save the data cube.
    overwrite : bool (default: False)
        Overwrite destination file if set to true.
    '''
    hdulist = fits.HDUList([fits.PrimaryHDU(data)])
    hdulist.writeto(filename, overwrite=overwrite)

def roll_2d(a, shift_i=None, shift_j=None):
    ''' Roll a 2D array along its axes.

    (A wrapper around np.roll, for 2D array.)

    Parameters
    ==========
    array : array_like (ndim >= 2)
        Input array.
    shift_i : int or None
        The number of places by which elements are shifted along axis 0.
        If None, shift by half the length of this axis.
    shift_j : int or None
        The number of places by which elements are shifted along axis 1.
        If None, shift by half the length of this axis.

    Returns
    =======
    output : ndarray
        Array with the same shape as the input array.

    Example
    =======
    >>> a = np.arange(9).reshape((3,3))
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> roll_2d(a, 1, 1)
    array([[8, 6, 7],
           [2, 0, 1],
           [5, 3, 4]])

    '''
    if shift_i is None:
        shift_i = a.shape[0] // 2
    if shift_j is None:
        shift_j = a.shape[1] // 2
    a = np.roll(a, shift_i, axis=0)
    a = np.roll(a, shift_j, axis=1)
    return a

def gauss2d(nx, ny, sx, sy, normalized=True):
    ''' Returns an array containing a 2D gaussian at its centre.

    Parameters
    ==========
    nx, ny : int
        The x and y size of the array.
    sx, sy : float
        The standard deviation of the gaussian along the x and y axes.
    normalized : bool (default: True)
        If True, generate a gaussian of integral 1. (*Warning*: this does not
            necessarily imply that arr.sum() == 1, unless nx and ny are large
            compared to sx and sy, respectively.)
        If False, the peak of the gaussian is set to 1.

    Returns
    =======
    arr : ndarray, shape (ny, nx)
        An array containing the gaussian data
    '''
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    arr = np.exp( - (x-nx/2)**2 / (2*sx**2) - (y-ny/2)**2 / (2*sy**2) )
    if normalized:
        arr /= 2 * np.pi * sx * sy
    return arr

def convolve_with_filter(arr, f):
    ''' Convolve a 2D array with a filter, using multiplication in the Fourier
    space.

    Parameters
    ==========
    arr : ndarray, shape (ni, nj)
        Data values.
    f : ndarray, shape (ni, nj)
        Filter values, to be convolved with the data.

    Returns
    =======
    conv : ndarray, shape (ni, nj)
        The convolution of arr with f.
    '''
    conv = np.fft.ifft2(
        np.fft.fft2(arr) * \
        np.fft.fft2(f)
        )
    conv = roll_2d(conv.real)
    return conv

def unsharp_mask(arr, radius=10, strength=1):
    ''' Apply unsharp masking to an array.

    This filters-out low spatial frequencies from an image. A blurred version
    of the image is first computed using a gaussian filter of standard
    deviation `radius`. This blurred image is then scaled by the factor
    `strength`, and removed from the original image.

    Parameters
    ==========
    arr : array, 2D
        Data values.
    radius : float (default: 10)
        Radius of the gaussian filter.
    strength : float (default: 1)
        Unsharp strength.

    Returns
    =======
    filtered_arr : array, 2D
        Filtered data values.
    '''
    ni, nj = arr.shape
    smooth_arr = convolve_with_filter(arr, gauss2d(nj, ni, radius, radius))
    filtered_arr = arr - smooth_arr * strength
    return filtered_arr
