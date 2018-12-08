************
Convolvables
************

Prysm features a rich implemention of Linear Shift Invariant (LSI) system theory.  Under this mathematical ideal, the transfer function is the product of the Fourier transform of a cascade of components, and the spatial distribution of intensity is the convolution of a cascade of components.  These features are usually used to blur objects or images with Point Spread Functions (PSFs), or model the transfer function of an opto-electronic system.  Within prysm there is a class :code:`Convolvable` which objects and PSFs inherit from.  You should rarely need to use the base class, except when subclassing it with your own models or objects.

>>> from prysm.convolution import Convolvable

The built-in convolvable objects are Slits, Pinholes, Tilted Squares, and Siemens Stars.  There are also two components, PixelAperture and OLPF, used for system modeling.

>>> from prysm import Slit, Pinhole, TiltedSquare, SiemensStar, PixelAperture, OLPF

Each is initialized with object-specific parameters,

>>> s = Slit(width=1, orientation='crossed')  # diameter, um
>>> p = Pinhole(width=1)
>>> t = TiltedSquare(angle=8, background='white', sample_spacing=0.05, samples=256))  # degrees
>>> star = SiemensStar(num_spokes=32, sinusoidal=False, background='white', sample_spacing=0.05, samples=256)
>>> pa = PixelAperture(width_x=5)  # diameter, um
>>> ol = OLPF(width_x=5*0.66)

Objects that take a background parameter will be black-on-white for :code:`background=white`, or white-on-black for :code:`background=black`.  Two objects are convolved via the :code:`conv` method, which returns :code:`self` on a new :code:`Convolvable` instance and is chainable,

>>> monstrosity = s.conv(p).conv(t).conv(star).conv(pa).conv(ol)

Some models require sample spacing and samples parameters while others do not.  This is because prysm has many methods of executing an FFT-based Fourier domain convolution under the hood.  If an object has a known analytical Fourier transform, the class has a method :code:`(Convolvable).analytic_ft` which has abscissa units of reciprocal microns.  If the analytic FT is present, it is used in lieu of numerical data.  Models that have analytical Fourier transforms also accept sample_spacing and samples parameters, which are used to define a grid in the spatial domain.  If two objects with analytical Fourier transforms are convolved, the output grid will have the finer sample spacing of the two inputs, and the larger span or window width of the two inputs.

The Convolvable constructor takes only four parameters,

>>> import numpy as np
>>> x = y = np.linspace(-20,20,256)
>>> z = np.random.uniform((256,256))
>>> c = Convolvable(data=z, unit_x=x, unit_y=y, has_analytic_ft=False)

:code:`has_analytic_ft` has a default value of :code:`False`.

Minimal labor is required to subclass :code:`Convolvable`.  For example, the :code:`Pinhole` implemention is simply:

.. code-block:: python

    class Pinhole(Convolvable):
        def __init__(self, width, sample_spacing=0.025, samples=0):
            self.width = width

            # produce coordinate arrays
            if samples > 0:
                ext = samples / 2 * sample_spacing
                x, y = m.linspace(-ext, ext, samples), m.linspace(-ext, ext, samples)
                xv, yv = m.meshgrid(x, y)
                w = width / 2
                # paint a circle on a black background
                arr = m.zeros((samples, samples))
                arr[m.sqrt(xv**2 + yv**2) < w] = 1
            else:
                arr, x, y = None, m.zeros(2), m.zeros(2)

            super().__init__(data=arr, unit_x=x, unit_y=y, has_analytic_ft=True)

        def analytic_ft(self, unit_x, unit_y):
            xq, yq = m.meshgrid(unit_x, unit_y)
            # factor of pi corrects for jinc being modulo pi
            # factor of 2 converts radius to diameter
            rho = m.sqrt(xq**2 + yq**2) * self.width * 2 * m.pi
            return m.jinc(rho).astype(config.precision)

which is less than 20 lines long.

:code:`Convolvable` objects have a few convenience properties and methods.  :code:`(Convolvable).slice_x` and its y variant exist and behave the same as slices on subclasses of :code:`OpticalPhase` such as :code:`Pupil`.  :code:`(Convolvable).plot_slice_xy` also works the same way.  :code:`(Convolvable).shape` is a convenience wrapper for :code:`(Convolvable).data.shape`, an :code:`(Convolvable).support_x, .support_y,` an :code:`.support` mimic the equivalent :code:`diameter` properties on :code:`OpticalPhase` inheritants.

:code:`(Convolvable).show` and :code:`(Convolvable).show_fourier` behave the same way as :code:`plot2d` methods found throughout prysm, except there are :code:`xlim` and :code:`ylim` parameters, which may be either single values, taken to be symmetric axis limits, or length-2 iterables of lower and upper limits.

Finally, :code:`Convolvable` objects may be initialized from images,

>>> c = Convolvable.from_file(path_to_your_image, scale=1)  # plate scale in um

and written out as 8-bit images,

>>> p = 'foo.png'  # or jpg, any format imageio can handle
>>> c.save(save_path)

In practical use, one will generally only use the :code:`conv`, :code:`from_file`, and :code:`save` methods with any degree of regularity.  The complete API documentation is below.  Attention should be paid to the docstring of :code:`conv`, as it describes some of the details associated with convolutions in prysm, their accuracy, and when they are used.

----

.. autoclass:: prysm.convolution.Convolvable
    :members:
    :undoc-members:
    :show-inheritance:

