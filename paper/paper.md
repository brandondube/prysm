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
  - name: Robert Sare
    orcid: 0000-0002-3837-605X
    affiliation: 1
affiliations:
  - name: Retro Refractions, LLC
    index: 1
date: 11 October 2018
bibliography: paper.bib
---

# Summary

``prysm`` is an open-source library for physical and first-order modeling of optical systems and analysis of related data. It is an unaffiliated sister library to ``PROPER`` and ``POPPY``, codes developed to do physical optics modeling for primarily space-based systems. Prysm has a more restrictive capability in that domain -- notably lacking multi-plane diffraction propagation -- but also offers a broader set of features.

The library supports the modeling and evaluation of an optical system at any level, in the pupil plane, image plane, or k-space (MTF).  This can also be done at the optical, or opto-electronic level by including detector elements.  Optical propgations are handled using the paraxial approximation of the Fresnel Transform via an FFT implementation.

`prysm` also features an `io` submodule for loading data from commercial and open source instrumentation and software into simple python structures (dicts, etc) or static methods on its classes for loading directly into `prysm`'s object system.  Notably, this support includes the most popular interferometers and MTF benches in the commercial marketplace.

These capabilities serve as the backbone of user programs for performing tasks such as image simulation, wavefront sensing, or robust analysis of metrology data utilizing cutting-edge methods from the literature.

Care has been given to the speed of calculation, and where possible `prysm` is able to leverage `numba` and `cupy` for accelerated calculations on the CPU and GPU, respectively.

# Acknowledgements

We would like to thank the help of many people throughout the development of prysm, most notably the members of Dr. James Fienup's research group at the Institute of Optics, University of Rochester, and Frans van den Bergh of CSIR.

# References
