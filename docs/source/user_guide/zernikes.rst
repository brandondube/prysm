Prysm supports two flavors of Zernike polynomials; the Fringe set up to the 49th term, and the Zemax Standard set up to the 48th term.  They have identical interfaces, so only one will be shown here.

Zernike notations are a subclass of :class:`Pupil`, so they support the same arguments to :code:`__init__`;

>>> from prysm import FringeZernike, StandardZernike
>>> p = FringeZernike(samples=123, epd=456.7, wavelength=1.0, opd_unit='nm', mask='dodecagon')

There are additional keyword arguments for each term, and the base (0 or 1) can be supplied; with base 0, the terms start at Z0 and range to Z48.  With base 1, they start at Z1 and range to Z49 (or Z48, for Standard Zernikes).  The Fringe set can also be used with unit RMS amplitude via the :code:`rms_norm` keyword argument.  Both notations also have nice print statements.

>>> p2 = FringeZernike(Z1=1, Z9=1, Z48=1, base=0, rms_norm=True)
>>> print(p2)
rms normalized Fringe Zernike description with:
        +1.000 Z1 - Tip (Y)
        +1.000 Z9 - Primary Trefoil Y
        +1.000 Z48 - Quinternary Spherical
        13.300 PV, 1.722 RMS

Notice that the RMS value is equal to sqrt(1^2 + 1^2 + 1^2) = sqrt(3) = 1.732 ~= 1.722.  The difference of ~1% is due to the array sizes used by prysm by default, if increased, e.g. by adding :code:`samples=1204` to the above :code:`p2 = FringeZernike(...)` line, the value would converge to the analytical one.

See :doc:`pupils <./pupils>` for information about general pupil functions.
