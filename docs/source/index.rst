*****
prysm
*****
:Release: |release|
:Date: |today|

prysm is an open-source library for physical and first-order modeling of optical systems and analysis of related data.  It is an unaffiliated sister library to PROPER and POPPY, codes developed to do physical optics modeling for primarily space-based systems.  Prysm has a more restrictive capability in that domain, notably lacking multi-plane diffraction propagation, but also offers a broader set of features.

.. contents::

Use Cases
---------
prysm aims to be a swiss army knife for optical engineers and students.  Its primary use cases include:

* Analysis of optical data
* robust numerical modeling of optical and opto-electronic systems based on physical optics
* wavefront sensing

prysm is on pypi:

>>> pip install prysm

prysm requires only `numpy <http://www.numpy.org/>`_ and `scipy <https://www.scipy.org/>`_.

To use an nVidia GPU, you must have `cupy <https://cupy.chainer.org/>`_ installed.  Plotting uses `matplotlib <https://matplotlib.org/>`_.  Images are read and written with `imageio <https://imageio.github.io/>`_.  Some MTF utilities utilize `pandas <https://pandas.pydata.org/>`_.  Reading of Zygo datx files requires `h5py <https://www.h5py.org/>`_.  Installation of these must be done offline.

User's Guide
------------

.. toctree::

   user_guide/index.rst


Examples
--------

.. toctree::

    examples/index.rst


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
