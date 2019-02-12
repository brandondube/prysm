"""Basic class holding data, used to recycle code."""


class BasicData:
    """Abstract base class holding some data properties."""
    _data_attr = 'data'

    @property
    def shape(self):
        """Proxy to data shape."""
        return getattr(self, self._data_attr).shape

    @property
    def size(self):
        """Proxy to data size."""
        return getattr(self, self._data_attr).size

    @property
    def samples_x(self):
        """Number of samples in the x dimension."""
        return self.shape[1]

    @property
    def samples_y(self):
        """Number of samples in the y dimension."""
        return self.shape[0]

    @property
    def sample_spacing(self):
        """center-to-center sample spacing."""
        return self.unit_x[1] - self.unit_x[0]

    @property
    def center_x(self):
        """Center "pixel" in x."""
        return self.samples_x // 2

    @property
    def center_y(self):
        """Center "pixel" in y."""
        return self.samples_y // 2

    @property
    def slice_x(self):
        """Retrieve a slice through the X axis of the phase.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase : `numpy.ndarray`

        """
        return self.unit_x, getattr(self, self._data_attr)[self.center_y, :]

    @property
    def slice_y(self):
        """Retrieve a slice through the Y axis of the phase.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase : `numpy.ndarray`

        """
        return self.unit_y, getattr(self, self._data_attr)[:, self.center_x]
