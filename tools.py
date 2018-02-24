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
