
# Align images

Collection of tools to align series of images containing the same subject, such
as astronomical data. :milky_way:

## Installation

Clone the repository (or download the sources), and install with pip:

~~~
git clone https://github.com/gpelouze/align_images
cd align_images
pip install .
~~~


## Usage

### Align a series of images

The alignment is done using two different functions in `align`:

- `align.compute_shifts()` returns the spatial shift for all images in the
  series. The shifts are determined by searching for the maximum of the 2D
  cross-correlation of these these images, relatively to either a reference
  image, or to all other image in the series.
- `align.align_cube()` builds a new series of aligned images using the output
  of `compute_shifts()`. Input images are interpolated to build the aligned
  ones.

The slowest steps of both functions (*ie.* the determination cross-correlation
maximum location and the interpolations) can to be parallelised by using the
keyword `processes`.

The function `align.roll_cube()` is provided as a faster but much less accurate
alternative to `align.align_cube()`. Make sure to understand its limitations
before using it!

See the functions documentation for more informations.

**Example**

~~~python
from astropy.io import fits
from align_images import align, tools

cube = fits.open('data.fits')[0].data
shifts = align.compute_shifts(cube, ref_frame=cube[0])
aligned_cube = align.align_cube(cube, shifts, processes=4)
tools.save_fits_cube(aligned_cube, 'aligned_data.fits')
~~~

### Determine the shift between two images

The shift between two images is determined computing their 2D
cross-correlation (CC), and finding the location its maximum. This is performed by
`align.track()`, which calls the appropriate function from `cc2d`, depending on
the required method.

Currently, 3 methods are provided by `cc2d` for computing the (CC), each
implementing different boundary conditions:

- `explicit()`: multiplication in the real space.
- `dft()`: multiplication in the real Fourier space.
- `scipy()`: a wrapper around `scipy.signal.correlate2d`.

While `explicit(boundary='drop')` is far less sensitive to edge effects than
`dft()`, it is also much slower. If a full CC map is not required,
`explicit_minimize()` can instead be used to locate the CC maximum within a
reasonable computation time.

**Normalization**

For any method, let `img1` and `img2` the entry images. We first subtract their
respective averages:  
*I* = `img1` - avg(`img1`),  
*J* = `img2` - avg(`img2`).

Then compute the normalisation factor, which is the product of the standard
deviations of *I* and *J*:  
*norm* = σ(*I*) × σ(*J*) = sqrt(sum(*I²*) × sum(*J²*)).

The cross-correlation returned by the method is normalized by this factor, such
that its maximum is 1:  
*cc = I ⋆ J / norm*.


### Miscellaneous tools

The submodule `tools` contains various functions either required by `align` and
`cc2d`, or that can be used to manipulate the images before they are aligned.


## License

Copyright (c) 2018 Gabriel Pelouze

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
