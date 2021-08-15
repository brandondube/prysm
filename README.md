# Prysm

[![CircleCI](https://circleci.com/gh/brandondube/prysm.svg?style=svg)](https://circleci.com/gh/brandondube/prysm?branch=master)
[![Documentation Status](https://readthedocs.org/projects/prysm/badge/?version=stable)](http://prysm.readthedocs.io/en/stable/?badge=stable)
[![Coverage Status](https://coveralls.io/repos/github/brandondube/prysm/badge.svg?branch=master)](https://coveralls.io/github/brandondube/prysm?branch=master) [![DOI](http://joss.theoj.org/papers/10.21105/joss.01352/status.svg)](https://doi.org/10.21105/joss.01352)


Prysm is a python 3.6+ library for numerical optics.  It contains features that are a superset of POPPY or PROPER for physical optics, as well as thin lens, thin film, and detector modeling.  There is also a submodule that can replace the software that comes with an interferometer for data analysis.  On CPU, end-to-end calculation is more than 100x as fast as the above for like-for-like calculations.  On GPU, prysm is more than 1,000x faster than its competition.

The library can be used for everything from forward modeling of optical systems from camera lenses to coronographs to reverse modeling and phase retrieval.  Due to its composable structure, it plays well with others and can be substituted in or out of other code easily.  For a list of features, see the documentation.  Of special note is prysm's interchangeable backend system, which allows the user to freely exchange numpy for cupy, enabling use of a GPU for _all_ computations, or other similar exchanges, such as pytorch for algorithmic differentiation.

## Installation

prysm is on pypi:
```
pip install prysm
```

prysm requires only [numpy](http://www.numpy.org/), and [scipy](https://www.scipy.org/).

### Optional Dependencies

Prysm uses numpy for array operations or any compatible library.  To use GPUs, you may install [cupy](https://cupy.chainer.org/) and use it as the backend at runtime.  Plotting uses [matplotlib](https://matplotlib.org/).  Images are read and written with [imageio](https://imageio.github.io/).  Some MTF utilities utilize [pandas](https://pandas.pydata.org/) and [seaborn](https://seaborn.pydata.org/).  Reading of Zygo datx files requires [h5py](https://www.h5py.org/).

## Features

### Propagation
- Fraunhofer, FFT or Matrix DFT
- Fresnel

### Polynomials
- Zernike
- Legendre
- Chebyshev
- Jacobi
- 2D-Q, Qbfs, Qcon
- Hopkins
- fitting
- projection

### Pupil Masks
- circles, binary and anti-aliased
- ellipses
- rectangles
- N-sided regular convex polygons
- N-vaned spiders

### Segmented systems
- parametrized pupil mask generation
- per-segment errors
- segment indexing / identification

### Image Simulation
- Convolution
- Smear
- Jitter
- in-the-box targets
- - Siemens' Star
- - Slanted Edge
- - BMW Target (crossed edges)
- - Pinhole
- - Slit
- - Tilted Square

### Metrics
- Strehl
- Encircled Energy
- RMS, PV, Sa, Std, Var
- Centroid
- FWHM, 1/e, 1/e^2
- PSD
- MTF / PTF / OTF
- PSD (and parametric fit, synthesis from parameters)
- slope / gradient
- Total integrated scatter
- Bandlimited RMS

### Detectors
- fully integrated noise model (shot, read, prnu, etc)
- arbitrary pixel apertures (square, oblong, purely numerical)
- optical low pass filters
- Bayer compositing, demosaicing

### Thin Films
- r, t parameters
- Brewster's angle
- Critical Angle
- Snell's law

### Refractive Index
- Cauchy's equation
- Sellmeier's equation

### Thin Lenses
- Defocus to delta z at the image and reverse
- object/image distance relation
- image/object distances and magnification
- image/object distances and NA/F#
- magnification and working F/#
- two lens BFL, EFL (thick lenses)

### Tilted Planes and other surfaces

- forward or reverse projection of surfaces such as those on Deformable Mirrors

Some features may be missing from this list.

### Interferometry

- PSD
- Low/High/Bandpass/Bandreject filtering
- spike clipping
- polynomial fitting and projection
- statistical evaluation (PV, RMS, PVr, Sa, bandlimited RMS...)
- total integrated scatter
- synthetic fringe maps with extra tilt fringes
- synthesize map from PSD spec

## Tutorials, How-Tos

See the [documentation](https://prysm.readthedocs.io/en/stable/tutorials/index.html) on [each](https://prysm.readthedocs.io/en/stable/how-tos/index.html)

## Contributing

If you find an issue with prysm, please open an [issue](https://github.com/brandondube/prysm/issues) or [pull request](https://github.com/brandondube/prysm/pulls).  Prysm has some usage of f-strings, so any code contributed is only expected to work on python 3.6+, and is licensed under the [MIT license](https://github.com/brandondube/prysm/blob/master/LICENSE.md).

Issue tracking, roadmaps, and project planning are done on Zenhub.  Contact Brandon for an invite if you would like to participate; all are welcome.

## Heritage

Organizations or projects using prysm:

- prysm was used to perform phase retrieval used to focus Nav and Hazcam, enhanced engineering cameras used to operate the Mars2020 Perserverence rover.

- prysm is used to build the official model of LOWFS, the Low Order Wavefront Sensing (and Control) system for the Roman coronoagraph instrument.  In this application, it has been used to validate dynamics of a hardware testbed to 35 picometers, or 0.08% of the injected dynamics.  The model runs at over 2kHz, faster than the real-time control system, at the same fidelity used to achieve 35 pm model agreement.

- prysm is used by several FFRDCs in the US, as well as their equivalent organizations abroad

- prysm is used by multiple ultra precision optics manufactures as part of their metrology data processing workflow

- prysm is used by multiple interferometer vendors to cross validate their own software offerings

- prysm is used at multiple universities to model optics both in a generic capacity and laboratory systems

There are likely many more.  These are key uses known to the authors.
