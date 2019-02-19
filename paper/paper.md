---
title: 'prysm: A Python optics module'
tags:
  - Python
  - optics
  - imaging
  - image simulation
  - diffraction
  - convolution
  - wavefront
  - interferogram
authors:
  - name: Brandon Dube
    orcid: 0000-0002-3837-605X
    affiliation: 1
affiliations:
  - name: Retro Refractions, LLC
    index: 1
date: 20 February 2019
bibliography: paper.bib
---

# Summary

``prysm`` is an open-source library for physical and first-order modeling of optical systems and analysis of related data.

The library supports the modeling, evaluation, and visualization of an optical system at any level, in the pupil plane, image plane, or k-space (MTF).  This can also be done at the optical, or opto-electronic system level by including detector elements.  Optical propgations are handled using the paraxial approximation of the Fresnel Transform via an FFT implementation.

`prysm` also features an `io` submodule for loading data from commercial and open source instrumentation and software into simple python structures (dicts, etc) or static methods on its classes for loading directly into `prysm`'s object system.  Notably, this support includes the most popular interferometers and MTF benches in the commercial marketplace.  Combined, these capabilities serve as the backbone of user programs supporting imaging system analysis by performing tasks such as image simulation, wavefront sensing, or robust analysis of metrology data utilizing cutting-edge methods from the literature.

The library is available for Linux, MacOS, and Windows and only carries core dependencies on numpy (T. E. Oliphant, 2006) and scipy (Jones, Oliphant, Peterson & others, 2001).  It will utilize a wide array of optional dependencies for some functionality.  For performance, prysm can leverage numba (Lam, Kwan, Pitrou, & Seibert, 2015) for acceleration of calculations on CPUs or cupy (Ryosuke, Unno, Nishino, Hido, & Loomis, 2017) on GPUs.  Plotting is implemented using matplotlib (Hunter, 2007), images are read and written using imageio (Klein et al, 2019) and deconvolved with point spread functions using scikit-image (Walt et al, 2014).  Some tabular formats for MTF data require pandas (Mckinney, 2010).

# Acknowledgements

We would like to thank the help of many people throughout the development of prysm, most notably the members of Dr. James Fienup's research group at the Institute of Optics, University of Rochester, and Frans van den Bergh of CSIR.

# References
