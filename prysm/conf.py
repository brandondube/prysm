"""Configuration for this instance of prysm."""
import copy

import numpy as np

from astropy import units as u

all_ap_unit_Types = (u.Unit, u.core.IrreducibleUnit, u.core.CompositeUnit)

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


def sanitize_unit(unit, existing_units):
    """Sanitize a unit token, either an astropy unit or a string.

    Parameters
    ----------
    unit : `astropy.Unit` or `str`
        unit or string version of unit
    existing_units : `Units`
        an existing unit, which stores a wavelength instance

    Returns
    -------
    `astropy.Unit`
        an astropy unit

    """
    if not isinstance(unit, all_ap_unit_Types):
        if unit.lower() in ('waves', 'wave', 'λ'):
            unit = existing_units.wavelength
        else:
            unit = getattr(u, unit)
    else:
        unit = unit

    return unit


def format_unit(unit_or_quantity, fmt):
    """(string) format a unit or quantity

    Parameters
    ----------
    unit_or_quantity : `astropy.units.Unit` or `astropy.units.Quantity`
        a unit or quantity
    fmt : `str`, {'latex', 'unicode'}
        a string format

    Returns
    -------
    `str`
        string

    """
    if isinstance(unit_or_quantity, all_ap_unit_Types):
        return unit_or_quantity.to_string(fmt)
    elif isinstance(unit_or_quantity, u.quantity.Quantity):
        return unit_or_quantity.unit.to_string(fmt)
    else:
        raise ValueError('must be a Unit or Quantity instance.')


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
        wavelength : `astropy.units` subclass, optional
            unit the wavelength is expressed in
        """
        if y is None:
            y = x
        self.x, self.y, self.z = x, y, z
        self.wavelength = wavelength

    def copy(self):
        return copy.deepcopy(self)


class Labels:
    """Labels holder for data instances."""
    def __init__(self, xy_base, z,
                 xy_additions, xy_addition_side='right',
                 addition_joiner=' ',
                 unit_prefix='[',
                 unit_suffix=']',
                 unit_joiner=' ',
                 show_units=True):
        """Create a new Labels instance

        Parameters
        ----------
        xy_base : `str`
            basic string used to build the X and Y labels
        z : `str`
            z label, stored as self._z to avoid clash with self.z()
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
        self.xy_base, self._z = xy_base, z
        self.xy_additions, self.xy_addition_side = xy_additions, xy_addition_side
        self.addition_joiner = addition_joiner
        self.unit_prefix, self.unit_suffix = unit_prefix, unit_suffix
        self.unit_joiner, self.show_units = unit_joiner, show_units

    def _label_factory(self, label, units):
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

            label_ = self.addition_joiner.join(label_basics)
        else:
            label_ = self._z

        unit_text = ''.join([self.unit_prefix,
                             format_unit(getattr(units, label), config.unit_format),
                             self.unit_suffix])
        label_ = self.unit_joiner.join([label_, unit_text])
        return label_

    def x(self, units):
        """X label."""
        return self._label_factory('x', units)

    def y(self, units):
        """Y label."""
        return self._label_factory('y', units)

    def z(self, units):
        """Z label."""
        return self._label_factory('z', units)

    def generic(self, units):
        """Generic label without extra X/Y annotation."""
        base = self.xy_base
        join = self.unit_joiner
        unit = format_unit(units.x, config.unit_format)
        prefix = self.unit_prefix
        suffix = self.unit_suffix
        return f'{base}{join}{prefix}{unit}{suffix}'

    def copy(self):
        return copy.deepcopy(self)


rel = u.def_unit(['rel'], format={'latex': 'Rel 1.0', 'unicode': 'Rel 1.0'})

HeNe = mkwvl(632.8, u.nm)

default_phase_units = Units(x=u.mm, y=u.mm, z=u.nm, wavelength=HeNe)
default_interferorgam_units = Units(x=u.pixel, y=u.pixel, z=u.nm, wavelength=HeNe)
default_image_units = Units(x=u.um, y=u.um, z=u.adu)
default_mtf_units = Units(x=u.mm ** -1, z=rel)
default_ptf_units = Units(x=u.mm ** -1, z=u.deg)

xi_eta = ['ξ', 'η']
x_y = ['X', 'Y']
default_pupil_labels = Labels(xy_base='Pupil', z='OPD', xy_additions=xi_eta)
default_interferogram_labels = Labels(xy_base='', z='Height', xy_additions=x_y)
default_convolvable_labels = Labels(xy_base='Image Plane', z='Irradiance', xy_additions=x_y)
default_mtf_labels = Labels(xy_base='Spatial Frequency', z='MTF', xy_additions=x_y)
default_ptf_labels = Labels(xy_base='Spatial Frequency', z='PTF', xy_additions=xi_eta)


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
                 unit_format='latex_inline',
                 show_units=True,
                 phase_units=default_phase_units,
                 image_units=default_image_units,
                 mtf_units=default_mtf_units,
                 ptf_units=default_ptf_units,
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
        self.unit_format = unit_format
        self.show_units = show_units
        self.phase_units, self.image_units = phase_units, image_units
        self.mtf_units, self.ptf_units = mtf_units, ptf_units
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
