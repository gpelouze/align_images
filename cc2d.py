#!/usr/bin/env python3

''' A set of methods for computing the cross-correlation of 2D images.

Currently, 3 methods are provided for computing the cross-correlation:

- explicit: multiplication in the real space.
- dft: multiplication in the real Fourier space.
- scipy: a wrapper around scipy.signal.correlate2d

For any method, let img1 and img2 the entry images. We first subtract their
respective averages:

    I = img1 - avg(img1)   and   J = img2 - avg(img2)

Then compute the normalisation factor, which is the product of the standard
deviations of I and J:

    norm = σI × σJ = sqrt(sum(I²) × sum(J²))

The cross-correlation returned by the method is:

    cc = I ⋆ J / norm

'''

import numpy as np
import scipy.signal as ss

from . import tools

def _prep_for_cc(img1, img2, inplace=False):
    ''' Prepare img1 and img2 for cross correlation computation:
    - set average to 0
    - fill masked values with 0 (if masked array)
    - compute the normalisation value

    Parameters
    ==========
    img1, img2 : ndarray or masked array
        The 2D arrays to prepare
    inplace : bool (default: False)
        If True, don't copy the arrays before removing the average.
        This saves time.

    Returns
    =======
    img1, img2 : ndarray or masked array
        The 2D arrays prepared
    norm : float
        The normalisation for these arrays
    '''
    if not inplace:
        a1 = img1.copy()
        a2 = img2.copy()
    else:
        a1 = img1
        a2 = img2
    a1 -= a1.mean()
    a2 -= a2.mean()
    try:
        a1 = a1.filled(0) # = fill with average
        a2 = a2.filled(0)
    except AttributeError:
        # not a masked array
        pass

    norm = np.sqrt(np.sum(a1**2) * np.sum(a2**2))

    return a1, a2, norm

def _get_padding_slice(img):
    ''' Get the slice for padding imag in `dft(... boundary='fill')`.

    Parameters
    ==========
    img : ndarray
        The 2D image.

    Returns
    =======
    s : slice
        The slice of the new array where the old data should be inserted and
        retrieved.
    N : tuple
        The size of the new array.
    '''
    n = np.array(img.shape)
    N = 2**np.ceil(np.log2(n * 2))
    N = N.astype(int)
    im = np.zeros(N[0])
    nmin = N//2 - n//2
    nmax = N//2 + n//2 + n%2
    s = (slice(nmin[0], nmax[0]), slice(nmin[1], nmax[1]))
    return s, N

def _pad_array(arr, s, N, pad_value=0):
    ''' Insert arr in a larger array of shape N at the position defined by s, a
    slice in the larger array. The area of the new array that don't contain
    data of arr are filled with pad_value.

    Parameters
    ==========
    arr : ndarray
        The array to insert in a larger array
    s : slice
        The slice of the larger array where the data from arr are to be
        inserted. This slice must have the same shape as arr.
    N : tuple
        The shape of the new larger array.
    pad_value : float
        The value used to fill the areas of the larger array that are outside
        of slice s.

    Return
    ======
    a : ndarray
        A larger array containing the values of arr at the positions defined by
        slice s.
    '''
    a = np.zeros(N) + pad_value
    a[s] = arr
    return a

def _unpad_array(arr, s, roll=False):
    ''' Reverse the operation performed by `_pad_array`.

    Parameters
    ==========
    arr : ndarray
        The larger array containing the padded data.
    s : slice
        The slice of the larger array where the data from arr are to be
        inserted. This slice must have the same shape as arr.
    roll : bool (default: False)
        A roll of half the size of the array is required before using the data.
        If True, roll, retrieve the data, and roll back.
    '''
    if roll:
        arr = tools.roll_2d(tools.roll_2d(arr)[s])
    else:
        arr = arr[s]
    return arr

def explicit_step(a1, a2, i, j, norm=None):
    ''' Compute the explicit cross-correlation between two arrays for a given
    shift.

    Parameters
    ==========
    a1, a2 : ndarray, 2D
        Data values.
    i, j : int
        The shift between a1 and a2 for which to compute the cross-correlation.
    norm : float or None (default: None)
        The value by which to normalize the result.
        If None, subtract their respective averages from the shifted version of
        a1 and a2:
            I = s_a1 - avg(s_a1); J = s_a2 - avg(s_a2),
        and compute a local norm:
            norm = sqrt(sum(I²) × sum(J²)).
        This is used to implement boundary='drop' when computing an explicit
        DFT map.

    Returns
    =======
    cc : float
        The cross-correlation of a1 with a2 for shift (i, j)
    '''
    nx, ny = a1.shape
    s1 = (
        slice(max(i, 0), min(nx+i-1, nx-1) + 1),
        slice(max(j, 0), min(ny+j-1, ny-1) + 1)
        )
    s2 = (
        slice(max(-i, 0), min(nx-i-1, nx-1) + 1),
        slice(max(-j, 0), min(ny-j-1, ny-1) + 1)
        )
    a1 = a1[s1]
    a2 = a2[s2]

    if norm is None:
        a1, a2, norm = _prep_for_cc(a1, a2, inplace=True)

    return np.sum(a1 * a2) / norm

def explicit(img1, img2, sxmax=None, symax=None, boundary='fill'):
    ''' Compute the cross-correlation of img1 and img2 using explicit
    multiplication in the real space.

    Parameters
    ==========
    img1, img2 : ndarray
    sxmax, symax : int or None (default: None)
        The maximum shift on the x and y axes resp. for which to compute the
        cross-correlation.
        If None, compute the cross correlation over the full shift domain.
    boundary : 'fill' or 'drop' (default: 'fill')
        How to handle boundary conditions. 'fill' is equivalent to padding the
        images with zeros. With 'drop' the cross-correlation is computing using
        only the overlapping part of the images.
    '''
    ny, nx = img1.shape
    if sxmax is None:
        sxmax = nx
    if symax is None:
        symax = ny

    if boundary == 'fill':
        img1, img2, norm = _prep_for_cc(img1, img2)
    elif boundary == 'drop':
        norm = None
    else:
        msg = "unexpected value for 'boundary': {}".format(boundary)
        raise ValueError(msg)

    cc = np.zeros((2 * sxmax + 1, 2 * symax + 1))
    for i in range(-sxmax, sxmax + 1):
        for j in range(-symax, symax + 1):
            cc[sxmax + i, symax + j] = explicit_step(
                img1, img2, i, j, norm=norm)

    cc = tools.roll_2d(cc)
    return cc

def dft(img1, img2, boundary='wrap'):
    ''' Compute the cross-correlation of img1 and img2 using multiplication in
    the Fourier space.

    Parameters
    ==========
    img1, img2 : ndarray
    boundary : str
        How to handle the boundary conditions.
        - 'wrap': perform a dumb product in the Fourier space, resulting in
          wrapped boundary conditions.
        - 'fill': the data are inserted in an array of size 2**n filled with
          zeros, where n is chosen such that the size of the new array is at
          least twice as big as the size of the original array.
    '''

    a1, a2, norm = _prep_for_cc(img1, img2)

    if boundary == 'fill':
        s, N = _get_padding_slice(img1)
        a1 = _pad_array(a1, s, N, pad_value=0)
        a2 = _pad_array(a2, s, N, pad_value=0)

        cc = np.fft.ifft2(
            np.conj(np.fft.fft2(a2)) * \
            np.fft.fft2(a1)
            )

        cc = _unpad_array(cc, s, roll=True)

    elif boundary == 'wrap':
        cc = np.fft.ifft2(
            np.conj(np.fft.fft2(a2)) * \
            np.fft.fft2(a1)
            )

    else:
        msg = "unexpected value for 'boundary': {}".format(boundary)
        raise ValueError(msg)

    cc /= norm

    return cc.real

def scipy(img1, img2, boundary='fill'):
    ''' Compute the cross-correlation of img1 and img2 using
    scipy.signal.correlate2d.

    Parameters
    ==========
    img1, img2 : ndarray
    boundary : str (default: 'fill')
        Passed to scipy.signal.correlate2d
    '''

    a1, a2, norm = _prep_for_cc(img1, img2)

    cc = ss.correlate2d(
        a1,
        a2,
        mode='same',
        boundary=boundary,
        fillvalue=0,
        )

    cc /= norm
    cc = tools.roll_2d(cc)
    return cc
