********
Zernikes
********

Prysm supports two flavors of Zernike polynomials; the Fringe set up to the 49th term, and the Zemax Standard set up to the 48th term.  They have identical interfaces, so only one will be shown here.

Zernike notations are a subclass of :class:`Pupil`, so they support the same arguments to :code:`__init__`;

>>> from prysm import FringeZernike, StandardZernike
>>> p = FringeZernike(samples=123, epd=456.7, wavelength=1.0, opd_unit='nm', mask='dodecagon')

There are additional keyword arguments for each term, and the base (0 or 1) can be supplied.  With base 0, the terms start at Z0 and range to Z48.  With base 1, they start at Z1 and range to Z49 (or Z48, for Standard Zernikes).  The Fringe set can also be used with unit RMS amplitude via the :code:`rms_norm` keyword argument.  Both notations also have nice print statements.

>>> p2 = FringeZernike(Z1=1, Z9=1, Z48=1, base=0, rms_norm=True)
>>> print(p2)
rms normalized Fringe Zernike description with:
        +1.000 Z1 - Tip (Y)
        +1.000 Z9 - Primary Trefoil Y
        +1.000 Z48 - Quinternary Spherical
        13.300 PV, 1.722 RMS

Notice that the RMS value is equal to sqrt(1^2 + 1^2 + 1^2) = sqrt(3) = 1.732 ~= 1.722.  The difference of ~1% is due to the array sizes used by prysm by default, if increased, e.g. by adding :code:`samples=1204` to the above :code:`p2 = FringeZernike(...)` line, the value would converge to the analytical one.

A Zernike pupil can also be initalized with an iterable (list) of coefficients,

>>> import numpy as np
>>> terms = np.random.rand(49)  # 49 zernike coefficients, e.g. from a wavefront sensor
>>> fz3 = FringeZernike(terms)

FringeZernike has many features StandardZernike does not.  At the module level are two functions,

>>> from prysm.fringezernike import fzname, fzset_to_magnitude_angle

:code:`fzname` takes an index and optional base (default 0) kwarg and returns the name of that term.  :code:`fzset_to_magnitude_angle` takes a non-sparse iterable of fringe zernike coefficients, starting with piston, and returns a dictionary with keywords of terms (e.g. "Primary Astigmatism") and items that are length 2 tuples of (magnitude, angle) where magnitude is in the same units as the zernike coefficients and the angle is in degrees.  :code:`fzset_to_magnitude_angle`'s output format can be seen below on the example of :code:`FringeZernike.magnitudes`.

:code:`FringeZernike` instances have a :code:`truncate` method which discards terms with indices higher than n.  For example,

>>> fz3.truncate(16)

this is less efficient, however, than simply slicing the coefficient vector,

>>> fz4 = FringeZernike(terms[:16])

and this slicing alternative should be used when one is sensitive to performance.

The top few terms may be extracted,

>>> fz4.top_n(5)
    [(0.9819642202790644, 5, 'Defocus'),
     (0.9591004803909723, 7, 'Primary Astigmatism 45°'),
     (0.9495975015239123, 28, 'Primary Pentafoil X'),
     (0.889828680566406, 8, 'Primary Coma Y'),
     (0.8831549729997366, 21, 'Secondary Trefoil X')]

or the terms listed by their pairwise magnitudes and clocking angles,

>>> fz4.magnitudes
    {'Piston': (0.7546621028832683, 90.0),
     'Tilt': (0.6603839752967117, 26.67150180654403),
     'Defocus': (0.14327809298942284, 90.0),
     'Primary Astigmatism': (0.7219964602989639, 85.19763587918983),
     'Primary Coma': (1.1351347586960325, 48.762867211696374),
     ...
     'Quinternary Spherical': (0.4974741523638292, 90.0)}

These things may be (bar) plotted;

>>> fz4.barplot(orientation='h', buffer=1, zorder=3)
>>> fz4.barplot_magnitudes(orientation='h', buffer=1, zorder=3)
>>> fz4.barplot_topn(n=5, orientation='h', buffer=1, zorder=3)

:code:`orientation` controls the axis on which the terms are enumerated.  :code:`h` results in vertical bars, :code:`v` is also accepted, as are :code:`horizontal` and :code:`vertical`.  :code:`buffer` is the number of terms' worth of spaces left on each side.  The default of 1 leaves a one bar margin.  :code:`zorder` is passed to matplotlib -- the default of 3 places the bars above any gridlines, which is an aesthetic choice.  Matplotlib has a general default of 1.

If you would like direct access to the underlying functions, there are two paths.  :code:`prysm._zernike` contains functions for the first 49 (Fringe ordered) zernike polynomials, for example

>>> from prysm._zernike import defocus

each of these takes arguments of (rho, phi).  They have names which end with :code:`_x`, or :code:`_00` and :code:`_45` for the terms which have that naming convention.

Perhaps more convenient is a dictionary which numbers them,

>>> from prysm._zernike import _zernikes
>>> from prysm.fringezernike import zernmap
>>> zfunction = _zernikes[zernmap[8]]
>>> zfunction, zfunction.name, zfunction.norm
    <function prysm._zernike.primary_spherical(rho, phi)>, 'Primary Spherical', 2.23606797749979

If these will always be evaluted over a square region enclosing the unit circle, a cache is available to speed computation,

>>> from prysm.fringezernike import fcache
>>> fcache(8, norm=True, samples=128)  # base 0
>>> fcache.clear()  # you should never have to do this unless you want to release memory

This cache instance is used internally by prysm, if you modify the returned arrays in-place, you probably won't like the result.  You can create your own independent instance,

>>> from prysm.fringezernike import FZCache
>>> my_fzcache = FZCache()

See :doc:`pupils <./pupils>` for information about general pupil functions.  Below, the Fringe type of Zernike description has its full documentation printed.

----

.. autoclass:: prysm.fringezernike.FringeZernike
    :members:
    :undoc-members:
    :show-inheritance:
