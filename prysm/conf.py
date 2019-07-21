"""Configuration for this instance of prysm."""
import numpy as np

from astropy import units as u


def mkwvl(quantity, base=u.um):
    """Generate a new Wavelength unit.

    Parameters
    ----------
    quantity : `float`
        number of (base) for the wavelength, e.g. quantity=632.8 with base=u.nm for HeNe.
    base : `astropy.units.Unit`
        base unit, e.g. um or nm

    Returns
    -------
    `astropy.units.Unit`
        new Unit for appropriate wavelength

    """
    return u.def_unit(['wave', 'wavelength'], quantity * base,
                      format={'latex': r'\lambda', 'unicode': 'λ'})


HeNe = mkwvl(632.8, u.nm)


class Units:
    """Units holder for data instances."""
    def __init__(self, x, z, y=None, wavelength=None):
        """Create a new Units instance

        Parameters
        ----------
        x : `astropy.units` subclass or `str`
            unit associated with the x coordinates
        z : `astropy.units` subclass or `str`
            unit associated with the z data
        y : `astropy.units` subclass or `str`, optional
            the same as x, copied from x if not given.
        wavelength : `astropy.units` subclass or `str`, optional
            unit the wavelength is expressed in

        """
        if not y:
            y = x
        self.x, self.y, self.z = x, y, z
        self.wavelength = wavelength


class Labels:
    """Labels holder for data instances."""
    def __init__(self, xybase, z, units, unit_formatter,
                 xy_additions, xy_addition_side,
                 addition_joiner,
                 unit_prefix,
                 unit_suffix,
                 unit_joiner,
                 show_units):
        """Create a new Labels instance

        Parameters
        ----------
        xybase : `str`
            basic string used to build the X and Y labels
        z : `str`
            z label
        units : `Units`
            units instance
        unit_formatter : `str`, optional
            formatter used by astropy.units.(unit).to_string
        xy_additions : iterable, optional
            text to add to the (x, y) labels
        xy_addition_side : {'left', 'right'. 'l', 'r'}, optional
            side to add the x and y additional text to, left or right
        addition_joiner : `str`, optional
            text used to join the x or y addition
        unit_prefix : `str`, optional
            prefix used to surround the unit text
        unit_suffix : `str`, optional
            suffix used to surround the unit text
        unit_joiner : `str`, optional
            text used to combine the base label and the unit
        show_units : `bool`, optional
            whether to print units
        """
        self.xybase, self.z = xybase, z
        self.units, self.unit_formatter = units, unit_formatter
        self.xy_additions, self.xy_addition_side = xy_additions, xy_addition_side
        self.addition_joiner = addition_joiner
        self.unit_prefix, self.unit_suffix = unit_prefix, unit_suffix
        self.unit_joiner, self.show_units = unit_joiner, show_units

    def _label_factory(self, label):
        """Factory method to produce complex labels.

        Parameters
        ----------
        label : `str`, {'x', 'y', 'z'}
            label to produce

        Returns
        -------
        `str`
            completed label

        """
        if label in ('x', 'y'):
            if label == 'x':
                xy_pos = 0
            else:
                xy_pos = 1
            label_basics = [self.xy_base]
            if self.xy_addition_side.lower() in ('left', 'l'):
                label_basics.insert(0, self.xy_additions[xy_pos])
            else:
                label_basics.append(self.xy_additions[xy_pos])

            label = self.addition_joiner.join(label_basics)
        else:
            label = self.z

        unit_text = ''.join([self.unit_prefix,
                             getattr(self.units, label).to_string(self.unit_formatter),
                             self.unit_suffix])
        label = self.unit_joiner.join([label, unit_text])
        return label

    @property
    def x(self):
        """X label."""
        return self._label_factory('x')

    @property
    def y(self):
        """Y label."""
        return self._label_factory('y')

    @property
    def z(self):
        """Z label."""
        return self._label_factory('z')


default_phase_units = Units(x=u.mm, y=u.mm, z=u.nm, wavelength=HeNe)
default_image_units = Units(x=u.um, y=u.um, z=u.adu)

default_phase_units = None
default_image_units = None

default_pupil_labels = None
default_interferogram_labels = None
default_convolvable_labels = None
default_mtf_labels = None
default_ptf_labels = None


class Config(object):
    """Global configuration of prysm."""
    def __init__(self,
                 precision=64,
                 backend=np,
                 zernike_base=1,
                 Q=2,
                 phase_cmap='inferno',
                 image_cmap='Greys_r',
                 lw=3,
                 zorder=3,
                 interpolation='lanczos',
                 unit_formatter='unicode',
                 xylabel_joiner=' ',
                 unit_prefix='[',
                 unit_suffix=']',
                 unit_joiner=', ',
                 show_units=True,
                 phase_units=default_phase_units,
                 image_units=default_image_units,
                 pupil_labels=default_pupil_labels,
                 interferogram_labels=default_interferogram_labels,
                 convolvable_labels=default_convolvable_labels,
                 mtf_labels=default_mtf_labels,
                 ptf_labels=default_ptf_labels):
        """Create a new `Config` object.

        Parameters
        ----------
        precision : `int`
            32 or 64, number of bits of precision
        backend : `str`, {'np'}
            a supported backend.  Current options are only "np" for numpy
        zernike_base : `int`, {0, 1}
            base for zernikes; start at 0 or 1
        Q : `float`
            oversampling parameter for numerical propagations
        phase_cmap : `str`
            colormap used for plotting optical phases
        image_cmap : `str`
            colormap used for plotting greyscale images
        lw : `float`
            linewidth
        zorder : `int`, optional
            zorder used for graphics made with matplotlib
        interpolation : `str`
            interpolation type for 2D plots
        unit_formatter : `str`, optional
            string passed to astropy.units.(unit).to_string
        xylabel_joiner : `str`, optional
            text used to glue together X/Y units and their basic string
        unit_prefix : `str`, optional
            text preceeding the unit's representation, after the joiner
        unit_suffix : `str`, optional
            text following the unit's representation
        unit_joiner : `str`, optional
            text used to glue basic labels and the units together
        show_units : `bool`, optional
            if True, shows units on graphics
        phase_units : `Units`
            default units used for phase-like types
        image_units : `Units`
            default units used for image-like types

        """
        self.initialized = False
        self.precision = precision
        self.backend = backend
        self.zernike_base = zernike_base
        self.chbackend_observers = []
        self.Q = Q
        self.phase_cmap = phase_cmap
        self.image_cmap = image_cmap
        self.lw = lw
        self.zorder = zorder
        self.interpolation = interpolation
        self.unit_formatter = unit_formatter
        self.xylabel_joiner = xylabel_joiner
        self.unit_prefix = unit_prefix
        self.unit_suffix = unit_suffix
        self.unit_joiner = unit_joiner
        self.show_units = show_units
        self.phase_units, self.image_units = phase_units, image_units
        self.pupil_labels = pupil_labels
        self.interferogram_labels = interferogram_labels
        self.convolvable_labels = convolvable_labels
        self.mtf_labels = mtf_labels
        self.ptf_labels = ptf_labels
        self.initialized = True

    @property
    def precision(self):
        """Precision used for computations.

        Returns
        -------
        `object` : `numpy.float32` or `numpy.float64`
            precision used

        """
        return self._precision

    @property
    def precision_complex(self):
        """Precision used for complex array computations.

        Returns
        -------
        `object` : `numpy.complex64` or `numpy.complex128`
            precision used for complex arrays

        """
        return self._precision_complex

    @precision.setter
    def precision(self, precision):
        """Adjust precision used by prysm.

        Parameters
        ----------
        precision : `int`, {32, 64}
            what precision to use; either 32 or 64 bits

        Raises
        ------
        ValueError
            if precision is not a valid option

        """
        if precision not in (32, 64):
            raise ValueError('invalid precision.  Precision should be 32 or 64.')

        if precision == 32:
            self._precision = np.float32
            self._precision_complex = np.complex64
        else:
            self._precision = np.float64
            self._precision_complex = np.complex128

    @property
    def backend(self):
        """Backend used.

        Returns
        -------
        `str`
            {'np'} only

        """
        return self._backend

    @backend.setter
    def backend(self, backend):
        """Set the backend used by prysm.

        Parameters
        ----------
        backend : `str`, {'np'}
            backend used for computations

        Raises
        ------
        ValueError
            invalid backend

        """
        if isinstance(backend, str):
            if backend.lower() in ('np', 'numpy'):
                backend = 'numpy'
            elif backend.lower() in ('cp', 'cu', 'cuda'):
                backend = 'cupy'

            exec(f'import {backend}')
            self._backend = eval(backend)
        else:
            self._backend = backend

        if self.initialized:
            for obs in self.chbackend_observers:
                obs(self._backend)

    @property
    def zernike_base(self):
        """Zernike base.

        Returns
        -------
        `int`
            {0, 1}

        """
        return self._zernike_base

    @zernike_base.setter
    def zernike_base(self, base):
        """Zernike base; base-0 or base-1.

        Parameters
        ----------
        base : `int`, {0, 1}
            first index of zernike polynomials

        Raises
        ------
        ValueError
            invalid base given

        """
        if base not in (0, 1):
            raise ValueError('By convention zernike base must be 0 or 1.')

        self._zernike_base = base


config = Config()
