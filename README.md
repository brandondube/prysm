# Prysm

[![Build Status](https://travis-ci.org/brandondube/prysm.svg?branch=master)](https://travis-ci.org/brandondube/prysm)
[![Documentation Status](https://readthedocs.org/projects/prysm/badge/?version=latest)](http://prysm.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/brandondube/prysm/badge.svg?branch=master)](https://coveralls.io/github/brandondube/prysm?branch=master)

A python3.6+ optics module.

## Installation

prysm is on pypi:
```
pip install prysm
```
For the most up-to-date version, you can install from github:
```
pip install git+git://github.com/brandondube/prysm.git
```

prysm requires [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [pandas](https://pandas.pydata.org/), and [matplotlib](https://matplotlib.org/).  Pip should take care of these for you, but if for some reason it doesn't, make sure they are installed first.

### Optional Dependencies

Prysm uses numpy for array operations.  If your environment has [numba](http://numba.pydata.org/) installed, it will automatically accelerate many of prysm's compuations.  To use an nVidia GPU, you must have [cupy](https://cupy.chainer.org/) installed.  The tests will require [pytest](http://pytest.org/) to run.

## Features

* Pupil modeling via:

* * Seidel notation

* * Fringe Zernike Polynomials, up to Z49 (base-1)

* * Zemax Standard Zernikes, up to Z48 (base-1)

* * Orthogonal and Orthonormal versions of both Zernike sets

* and with:

* * gaussian apodization

* * noncircular apertures

* * * n-sided regular polygons

* * * rotated ellipses

* * * user-provided masks

* Point Spread Function (PSF) models

* Modulation Transfer Function (MTF) models

* Detector models and sampling

* Optical Low Pass Filter (OLPF) models

* Image synthesis

* Tools for working with interferometric data

* Utilities for working with MTF data

* file readers for:

* * Trioptics data files

* * Zygo data files

* Thin lens / geometrical optics models

* Shack-Hartmann sensor models

* An Object-Oriented Lens model based on physical optics

## Usage

See [Examples](https://github.com/brandondube/prysm/tree/master/Examples) for jupyter notebooks with more complete samples.

### Model the pupil of a diffraction limited optical system, its PSF, and its MTF.
```python
from matplotlib import pyplot as plt
from prysm import Pupil, PSF, MTF, plot_fourier_chain

pupil = Pupil()
psf = PSF.from_pupil(pupil, efl=1)  # focal length in units of mm
mtf = MTF.from_psf(psf)

plot_fourier_chain(pupil, psf, mtf)
plt.show()
```

### Introduce some aberrations

```python
from prysm import FringeZernike

aberrated_pupil = FringeZernike(Z9=1)
pupil += aberrated_pupil
mtf2 = MTF.from_pupil(pupil, efl=1)  # can also skip PSF as long as EFL is given
fig, ax = mtf2.plot2d()
plt.show()
```
Notice that FringeZernike is also a type of pupil, and pupil objects support addition and subraction.  `MTF.from_pupil` could have been called wiith `aberrated_pupil` directly.

## Documentation

See [the docs](https://prysm.readthedocs.io/) -- the docstrings and [Examples](https://github.com/brandondube/prysm/tree/master/Examples) are much more up-to-date and probably more useful.  The docs are currently just an API reference and will require searching to even find anything within them.

## Contributing

If you find an issue with prysm, please open an [issue](https://github.com/brandondube/prysm/issues) or [pull request](https://github.com/brandondube/prysm/pulls).  Prysm has some usage of f-strings, so any code contributed is only expected to work on python 3.6+, and is licensed under the [MIT license](https://github.com/brandondube/prysm/blob/master/LICENSE.md).  The library is
most in need of contributions in the form of tests and documentation.
