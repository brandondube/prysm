# Prysm

[![Build Status](https://travis-ci.org/brandondube/prysm.svg?branch=master)](https://travis-ci.org/brandondube/prysm)
[![Documentation Status](https://readthedocs.org/projects/prysm/badge/?version=stable)](http://prysm.readthedocs.io/en/stable/?badge=stable)
[![Coverage Status](https://coveralls.io/repos/github/brandondube/prysm/badge.svg?branch=master)](https://coveralls.io/github/brandondube/prysm?branch=master) [![DOI](http://joss.theoj.org/papers/10.21105/joss.01352/status.svg)](https://doi.org/10.21105/joss.01352)


A python3.6+ module for physical optics based modeling and processing of data from commerical and open source instrumentation.

## Installation

prysm is on pypi:
```
pip install prysm
```

prysm requires only [numpy](http://www.numpy.org/) and [scipy](https://www.scipy.org/).

### Optional Dependencies

Prysm uses numpy for array operations.  If your environment has [numba](http://numba.pydata.org/) installed, it will automatically accelerate many of prysm's compuations.  To use an nVidia GPU, you must have [cupy](https://cupy.chainer.org/) installed.  Plotting uses [matplotlib](https://matplotlib.org/).  Images are read and written with [imageio](https://imageio.github.io/).  Some MTF utilities utilize [pandas](https://pandas.pydata.org/).  Reading of Zygo datx files requires [h5py](https://www.h5py.org/).

## Features

Prysm features robust tools for modeling and propagation of wavefronts to image planes and MTF.  It also features object synthesis routines and a flexible convolution system in support of image simulation.  Finally, it contains rich features for analysis of interferometric data.

For a complete list of features, see [the docs](https://prysm.readthedocs.io/en/stable/).

## Examples

Several [examples](https://prysm.readthedocs.io/en/stable/examples/index.html) are provided in the documentation.

## User's Guide

A [guide](https://prysm.readthedocs.io/en/stable/user_guide/index.html) for using the library is provided in the documentation.

## Contributing

If you find an issue with prysm, please open an [issue](https://github.com/brandondube/prysm/issues) or [pull request](https://github.com/brandondube/prysm/pulls).  Prysm has some usage of f-strings, so any code contributed is only expected to work on python 3.6+, and is licensed under the [MIT license](https://github.com/brandondube/prysm/blob/master/LICENSE.md).  The library is
most in need of contributions in the form of tests and documentation.
