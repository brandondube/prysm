*****
prysm
*****
:Release: |release|
:Date: |today|

prysm is an open-source library for physical and first-order modeling of optical systems and analysis of related data.  You can use prysm to...

* Do multi-plane diffraction calculations
* Do image chain or integrated modeling
* Process data from commercial interferometers, MTF benches, and design/analysis software


This list is not exhaustive, feel free to file a PR to add more to this list!

This documentation is divided into four categories; a series of tutorials that teach step-by-step, a set of how-tos that show individual more advanced usages, a reference guide that includes the API-level documentation, and a set of explanation articles that teach you the core philsophy and design behind this library.  If you're looking for "getting started" - take a look at tutorials!

.. contents::


Installation
------------

prysm is on pypi:

>>> pip install prysm

prysm requires only `numpy <http://www.numpy.org/>`_ and `scipy <https://www.scipy.org/>`_.

Optionally, plotting uses `matplotlib <https://matplotlib.org/>`_.  Images are read and written with `imageio <https://imageio.github.io/>`_.  Some MTF utilities utilize `pandas <https://pandas.pydata.org/>`_.  Reading of Zygo datx files requires `h5py <https://www.h5py.org/>`_.  Installation of these must be done prior to installing prysm.

Prysm's backend is runtime interchangeable, you may also install and use `cupy <https://cupy.chainer.org/>`_ or other numpy/scipy API compatible libraries if you wish.


Tutorials
---------

.. toctree::
    :maxdepth: 2

    tutorials/index.rst


How-Tos
-------

.. toctree::
    :maxdepth: 2

    how-tos/index.rst


Explanations (deep dives)
-------------------------

.. toctree::
    :maxdepth: 2

    explanation/index.rst

API Reference
-------------

.. toctree::
    :maxdepth: 2

    api/index.rst

Contributing
------------

.. toctree::

    contributing.rst

Release History
---------------

.. toctree::
    :maxdepth: 2

    releases/index.rst
