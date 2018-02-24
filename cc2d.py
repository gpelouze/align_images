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

def _prep_for_cc(img1, img2):
    ''' Prepare img1 and img2 for cross correlation computation:
    - set average to 0
    - fill masked values with 0 (if masked array)
    - compute the normalisation value

    Parameters
    ==========
    img1, img2 : ndarray or masked array
        The 2D arrays to prepare

    Returns
    =======
    img1, img2 : ndarray or masked array
        The 2D arrays prepared
    norm : float
        The normalisation for these arrays
    '''
    a1 = img1.copy()
    a2 = img2.copy()
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

def explicit(img1, img2, sxmax, symax):
    ''' Compute the cross-correlation of img1 and img2 using explicit
    multiplication in the real space.

    Parameters
    ==========
    img1, img2 : ndarray
    sxmax, symax : int
        The maximum shift on the x and y axes resp. for which to compute the
        cross-correlation.
    '''
    img1, img2, norm = _prep_for_cc(img1, img2)

    nx, ny = img1.shape
    cc = np.zeros((2 * sxmax + 1, 2 * symax + 1))

    for i in range(-sxmax, sxmax + 1):
        for j in range(-symax, symax + 1):

            s1 = (
                slice(max(i, 0), min(nx+i-1, nx-1) + 1),
                slice(max(j, 0), min(ny+j-1, ny-1) + 1)
                )
            s2 = (
                slice(max(-i, 0), min(nx-i-1, nx-1) + 1),
                slice(max(-j, 0), min(ny-j-1, ny-1) + 1)
                )

            a1 = img1[s1]
            a2 = img2[s2]

            cc_ij = np.sum(a1 * a2)
            cc[sxmax + i, symax + j] = cc_ij

    cc /= norm

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

    return cc
