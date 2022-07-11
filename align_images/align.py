#!/usr/bin/env python3

''' Collection of tools to align series of images.

The alignment is done using two functions:

- compute_shifts() returns the spatial shift for all images in the
  series. The shifts are determined by searching for the maximum of the 2D
  cross-correlation of these these images, relatively to either a reference
  image, or to all other image in the series.
- align_cube() builds a new series of aligned images using the output of
  compute_shifts(). Input images are interpolated to build the aligned ones.

The slowest steps of both functions (ie. the determination cross-correlation
maximum location and the interpolations) can to be parallelised by using the
keyword `processes`.

The function roll_cube() is provided as a faster but much less accurate
alternative to align_cube(). Make sure to understand its limitations before
using it!

'''

import functools
import multiprocessing as mp
import warnings

import numpy as np
import scipy.interpolate as si

from . import tools
from . import cc2d

def track(img1, img2,
        sub_px=True, missing=None, cc_function='dft', cc_boundary='wrap',
        return_full_cc=False,
        **kwargs):
    ''' Return the shift between img1 and img2 by computing their cross
    correlation.

    Parameters
    ==========
    img1, img2 : 2D ndarrays
        The images to correlate
    sub_px : bool (default: True)
        Whether to determine shifts with sub-pixel accuracy. Set to False for
        faster, less accurate results.
    missing : float or None (default: None)
        The value of the pixels in the image that should be considered as
        'missing', and thus discarded before computing the cross correlation.
        If set to None, don't handle missing values.
        If your missing values are 'None', you’re out of luck.
    cc_function : str (default: 'dft')
        Name of the function to use to compute the cross-correlation between
        two frames. Accepted values are 'dft', 'explicit', and 'scipy'.
        Functions of cc2d that have the same name are used.
    cc_boundary : str or None (default: None)
        If not None, pass this value as the `boundary` keyword to cc_function.
        See cc_function documentation for accepted values.
    return_full_cc : bool (default: False)
        If True, return the full cross-correlation array.
        If False, only return the maximum cross-correlation.
    **kwargs :
        Passed to cc_function.

    Returns
    =======
    offset : ndarray
        An array containing the optimal (y, x) offset between the input images
    cc : float or 3D array
        Depending on the value of return_full_cc, either the value of the
        cross-correlation at the optimal offset, or the full cross-correlation
        array.
    '''

    assert img1.shape == img2.shape, 'Images must have the same shape.'
    ny, nx = img1.shape

    sy, sx = ny // 2, nx // 2

    if missing is not None:
        if np.isnan(missing):
            mask1 = np.isnan(img1)
            mask2 = np.isnan(img2)
        else:
            mask1 = (img1 == missing)
            mask2 = (img2 == missing)

        img1 = np.ma.array(img1, mask=mask1)
        img2 = np.ma.array(img2, mask=mask2)

    cc_functions = {
        'dft': cc2d.dft,
        'explicit': cc2d.explicit,
        'scipy': cc2d.scipy,
        }
    cc_function = cc_functions[cc_function]
    if cc_boundary:
        cc = cc_function(img1, img2, boundary=cc_boundary, **kwargs)
    else:
        cc = cc_function(img1, img2, **kwargs)
    cc = tools.roll_2d(cc, shift_i=sy, shift_j=sx)

    maxcc = np.nanmax(cc)
    cy, cx = np.where(cc == maxcc)
    if not(len(cy) == 1 and len(cx) == 1):
        m = 'Could not find a unique cross correlation maximum.'
        warnings.warn(m, RuntimeWarning)
    cy = cy[0]
    cx = cx[0]

    offset = np.zeros((2))

    if (not sub_px) or (maxcc == 0):
        offset[0] = cy
        offset[1] = cx

    else:

        # parabolic interpolation about minimum:
        yi = [cy-1, cy, (cy+1) % ny]
        xi = [cx-1, cx, (cx+1) % nx]
        ccy2 = cc[yi, cx]**2
        ccx2 = cc[cy, xi]**2

        yn = ccy2[2] - ccy2[1]
        yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]
        xn = ccx2[2] - ccx2[1]
        xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]

        if yd != 0 and not np.isnan(yd):
            offset[0] = yi[2] - yn / yd - 0.5
        else:
            offset[0] = float(cy)

        if xd != 0 and not np.isnan(xd):
            offset[1] = xi[2] - xn / xd - 0.5
        else:
            offset[1] = float(cx)

    offset[0] = sy - offset[0]
    offset[1] = sx - offset[1]

    if return_full_cc:
        return offset, cc
    else:
        return offset, maxcc

def compute_shifts(cube, ref_frame=None, processes=1, **kwargs):
    ''' Compute the shifts between each frame of a cube.

    Parameters
    ==========
    cube : 3D ndarray
        The cube for which to compute the shifts.
    ref_frame : ndarray or None (default: None)
        - If ndarray, compute the shifts for each frame of the cube relatively
          to this reference frame only.
        - If None, compute the shifts of each frame of the cube relatively to all
          othe frames, then take the average of these shifts.
        The first method is faster (execution time prop. to the number of
        frames N) but less accurate. The second is much slower (prop. to N²),
        but more accurate.
    processes : int (default: 1)
        The number of processes to use.
    **kwargs : passed to track().

    Returns
    =======
    shifts : 2D ndarray
        Shifts for each frame of the cube, along both dimensions. Each row of
        the array corresponds to a frame in the cube. Column 0 contains x
        shifts, column 2 contains y shifts.
    '''

    if ref_frame is not None:
        # Compute shifts relatively to ref_frame
        print('Computing shifts...', end=' ')
        p = mp.Pool(processes)
        try:
            shifts = p.map(
                functools.partial(track, ref_frame, **kwargs),
                iter(cube),
                chunksize=1,
                )
        finally:
            p.terminate()
        print('Done.')
        shifts = np.array([s for s, max_cc in shifts]).T
        y_shifts, x_shifts = shifts

    else:
        # Compute shifts using all possible frame couples
        n_frames = cube.shape[0]
        x_matrix = np.zeros((n_frames, n_frames))
        y_matrix = np.zeros((n_frames, n_frames))
        max_cc = np.zeros((n_frames, n_frames))

        def _cube_coord_iter(n_frames):
            for i in range(n_frames - 1):
                for j in range(i + 1, n_frames):
                    yield i, j

        def _cube_iter(intensity_cube):
            n_frames = intensity_cube.shape[0]
            for i, j in _cube_coord_iter(n_frames):
                yield intensity_cube[i], intensity_cube[j]

        print('Computing shifts...', end=' ')
        p = mp.Pool(processes)
        try:
            corr_matrix = p.starmap(
                functools.partial(track, **kwargs),
                _cube_iter(cube),
                chunksize=1,
                )
        finally:
            p.terminate()
        print('Done.')

        for (i, j), (offset, maxi) in zip(
                _cube_coord_iter(n_frames),
                corr_matrix):
            y_matrix[i, j] = offset[0]
            y_matrix[j, i] = - offset[0]
            x_matrix[i, j] = offset[1]
            x_matrix[j, i] = - offset[1]
            max_cc[i, j] = maxi
            max_cc[j, i] = maxi

        x_shifts = np.average(x_matrix, weights=max_cc, axis=0)
        y_shifts = np.average(y_matrix, weights=max_cc, axis=0)

    sub_px = kwargs.get('sub_px', True)
    if not sub_px:
        x_shifts = x_shifts.astype(int)
        y_shifts = y_shifts.astype(int)

    shifts = np.stack((x_shifts, y_shifts)).T
    return shifts

def align_frame(frame, shift, points_old, grid_new, points_new, method):
    points_old = points_old - shift
    values_old = frame.flatten()

    values_new = si.griddata(
        points_old,
        values_old,
        points_new,
        method=method,
        )
    values_new = values_new.reshape(grid_new[0].shape[:2])

    return values_new

def align_cube(cube, shifts, method='nearest', ref_frame=None, processes=1):
    ''' Project a cube where each slice is shifted by a factor that needs to be
    corrected.

    The new cube size is chosen such that it covers all the area covered by
    all its individual slices.

    Parameters
    ==========
    cube : ndarray
        A 3D cube with dimensions (n, y, x) the data.
    shifts : tuple
        A tuple containing 2 arrays of shapes (x, y), each containing the
        correction to apply to the corresponding axis of the cube.
    method : str (default: 'nearest')
        Interpolation method, passed to scipy.interpolate.griddata().
    ref_frame : int or None (default: None)
        If not None, the cube is aligned relatively to the specified frame.
        If None, new coordinates are chosen such that all offset frames fit
        within the new cube.
    processes : int (default: 1)
        The number of processes to use.

    Returns
    =======
    new_cube : ndarray
        The corrected cube
    '''

    n_frames, ymax_old, xmax_old = cube.shape

    # chose x and y bounds such the new array covers all the area covered by
    # the shifted slices of the input array
    if ref_frame is None:
        xmin, ymin = - np.ceil(np.max(shifts, axis=0))
        xmax, ymax = - np.ceil(np.min(shifts, axis=0)) + (xmax_old, ymax_old)
        xmin, ymin = int(xmin), int(ymin)
        xmax, ymax = int(xmax), int(ymax)
    else:
        xmin, ymin = 0, 0
        xmax, ymax = xmax_old, ymax_old

    # prepare new and old points for si.griddata

    # prepare new points
    grid_new = np.meshgrid(
        np.arange(xmin, xmax),
        np.arange(ymin, ymax),
        )
    points_new = np.array([g.flatten() for g in grid_new]).T

    # shift shifts (:D) to make them relative to ref_frame
    if ref_frame is not None:
        shifts = [s - shifts[ref_frame] for s in shifts]

    # prepare old points adding a 1 px nan border to mitigate si.griddata with
    # method='nearest' behaviour
    grid_old = np.meshgrid(
        range(-1, xmax_old + 1),
        range(-1, ymax_old + 1),
        )
    points_old = np.array([g.flatten() for g in grid_old]).T

    # add 1 px nan border data to old cube
    border_cube_shape = np.array(cube.shape) + (0, 2, 2)
    border_cube = np.ones(border_cube_shape) * np.nan
    border_cube[:, 1:-1, 1:-1] = cube
    cube = border_cube

    # prepare new cube
    new_cube = np.zeros((n_frames, ) + grid_new[0].shape)

    def _cube_iter(cube, shifts, points_old, grid_new, points_new, method):
        for frame, shift in zip(cube, shifts):
            yield frame, shift, points_old, grid_new, points_new, method

    print('Aligning cube...', end=' ')
    p = mp.Pool(processes)
    try:
        new_cube = p.starmap(
            align_frame,
            _cube_iter(cube, shifts, points_old, grid_new, points_new, method),
            chunksize=1,
            )
    finally:
        p.terminate()
    print('Done.')

    new_cube = np.array(new_cube)

    return new_cube

def roll_cube(cube, shifts):
    ''' Roll each frames in a cube by an integer number of pixels

    This ‘aligns’ the cube very quickly, to the cost of significant edge
    effects. Eg. when shifting a frame by 10 pixels along the x axis, values
    from the 10 last columns are inserted in the 10 first columns.

    Parameters
    ==========
    cube : 3D ndarray (shape: n, x, y)
        A cube containing the data to roll.
    shifts : 2D ndarray (shape: n, 2)
        The shifts by which to roll the cube. Each row contains the x and y
        shifts for a given frame of the cube.

    Returns
    =======
    rolled_cube : ndarray
        The corrected cube
    '''
    rolled_cube = []
    shifts = shifts.astype(int)
    for frame, shift in zip(cube, shifts):
        sx, sy = shift
        rolled_cube.append(tools.roll_2d(frame, shift_i=-sy, shift_j=-sx))
    rolled_cube = np.array(rolled_cube)
    return rolled_cube
