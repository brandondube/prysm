"""Utility functions."""
from operator import itemgetter

from .mathops import engine as e


def mean(array):
    """Return the mean value of the valid elements of an array.

    Parameters
    ----------
    array : `numpy.ndarray`
        array of values

    Returns
    -------
    `float`
        mean value

    """
    non_nan = e.isfinite(array)
    return array[non_nan].mean()


def pv(array):
    """Return the PV value of the valid elements of an array.

    Parameters
    ----------
    array : `numpy.ndarray`
        array of values

    Returns
    -------
    `float`
        PV of the array

    """
    non_nan = e.isfinite(array)
    return array[non_nan].max() - array[non_nan].min()


def rms(array):
    """Return the RMS value of the valid elements of an array.

    Parameters
    ----------
    array : `numpy.ndarray`
        array of values

    Returns
    -------
    `float`
        RMS of the array

    """
    non_nan = e.isfinite(array)
    return e.sqrt((array[non_nan] ** 2).mean())


def Sa(array):
    """Return the Ra value for the valid elements of an array.

    Parameters
    ----------
    array: `numpy.ndarray`
        array of values

    Returns
    -------
    `float`
        Ra of the array

    """
    non_nan = e.isfinite(array)
    ary = array[non_nan]
    mean = ary.mean()
    return abs(ary - mean).sum() / ary.size


def std(array):
    """Return the standard deviation of the valid elements of an array.

    Parameters
    ----------
    array: `numpy.ndarray`
        array of values

    Returns
    -------
    `float`
        std of the array

    """
    non_nan = e.isfinite(array)
    ary = array[non_nan]
    return ary.std()


def guarantee_array(variable):
    """Guarantee that a varaible is a numpy ndarray and supports -, *, +, and other operators.

    Parameters
    ----------
    variable : `number` or `numpy.ndarray`
        variable to coalesce

    Returns
    -------
    `object`
        an object that  supports * / and other operations with ndarrays

    Raises
    ------
    ValueError
        non-numeric type

    """
    if type(variable) in [float, e.ndarray, e.int32, e.int64, e.float32, e.float64, e.complex64, e.complex128]:
        return variable
    elif type(variable) is int:
        return float(variable)
    elif type(variable) is list:
        return e.asarray(variable)
    else:
        raise ValueError(f'variable is of invalid type {type(variable)}')


def ecdf(x):
    """Compute the empirical cumulative distribution function of a dataset.

    Parameters
    ----------
    x : `iterable`
        Data

    Returns
    -------
    xs : `numpy.ndarray`
        sorted data
    ys : `numpy.ndarray`
        cumulative distribution function of the data

    """
    xs = e.sort(x)
    ys = e.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def sort_xy(x, y):
    """Sorts a pair of x and y iterables, returning arrays in order of ascending x.

    Parameters
    ----------
    x : `iterable`
        a list, numpy ndarray, or other iterable to sort by
    y : `iterable`
        a list, numpy ndarray, or other iterable that is y=f(x)

    Returns
    -------
    sorted_x : iterable
        an iterable containing the sorted x elements
    sorted_y : iterable
        an iterable containing the sorted y elements

    """
    # zip x and y, sort by the 0th element (x) of each tuple in zip()
    _ = sorted(zip(x, y), key=itemgetter(0))
    sorted_x, sorted_y = zip(*_)
    return sorted_x, sorted_y
