"""Cost functions, aka figures of merit for models."""
import functools
import numbers

from prysm.mathops import np


def _masked_cost(fn):
    """Wrap a cost function to handle masking and dtype validation.

    The wrapped fn(M, D) -> (cost, grad) is called on either the full
    arrays (when mask is None) or on the masked subset.  In the masked
    case the returned gradient is scattered back into a zero-init buffer
    shaped like the original M so the caller always sees a gradient
    aligned with the full input.

    Also asserts M.dtype == D.dtype when both have a dtype, to catch
    silent float32/float64 upcasts that waste compute and memory.

    """
    @functools.wraps(fn)
    def wrapper(M, D, mask=None):
        if hasattr(M, 'dtype') and hasattr(D, 'dtype') and M.dtype != D.dtype:
            raise TypeError(
                f"{fn.__name__}: input dtype mismatch — first array is "
                f"{M.dtype}, second is {D.dtype}; cast one to match before calling"
            )
        if mask is None:
            return fn(M, D)

        M_m = M[mask]
        D_m = D if isinstance(D, numbers.Number) else D[mask]
        cost, grad_m = fn(M_m, D_m)
        grad = np.zeros_like(M)
        grad[mask] = grad_m
        return cost, grad

    return wrapper


@_masked_cost
def bias_and_gain_invariant_error(I, D):  # NOQA
    """Bias and gain invariant error.

    This cost function computes internal least mean squares estimates of the
    overall bias (DC pedestal) and gain between the signal I and D.  This
    objective is useful when the overall signal level is ambiguous in phase
    retrieval type problems, and can significantly help stabilize the
    optimization process.

    See also: mean_square_error

    Parameters
    ----------
    I : ndarray
        'intensity' or model data, any float dtype, any shape
    D : ndarray
        'data' or true mesaurement to be matched, any float dtype, any shape
    mask : ndarray, optional
        logical array with elements to keep (True) or exclude (False)

    Returns
    -------
    float, ndarray
        cost, dcost/dI

    """
    Ihat = I - I.mean()
    Dhat = D - D.mean()

    N = I.size

    num = (Ihat * Dhat).sum()
    den = (Ihat * Ihat).sum()
    alpha = num / den

    alphaI = alpha * I

    beta = (D - alphaI) / N

    R = 1 / ((D * D).sum())
    raw_err = (alphaI + beta) - D
    err = R * (raw_err * raw_err).sum()
    # 2 is from raw_err squared; R and alpha are pass-throughs of the chain rule
    grad = 2 * R * alpha * raw_err
    return err, grad


@_masked_cost
def mean_square_error(M, D):
    """Mean square error.

    Parameters
    ----------
    M : ndarray
        "model data"
    D : ndarray
        "truth data"
    mask : ndarray, optional
        True where M should contribute to the cost, False where it should not

    Returns
    -------
    float, ndarray
        cost, dcost/dM

    """
    diff = M - D
    alpha = 1 / diff.size
    cost = (diff * diff).sum() * alpha
    grad = 2 * alpha * diff
    return cost, grad


@_masked_cost
def negative_loglikelihood(y, yhat):
    """Negative log likelihood.

    Parameters
    ----------
    y : ndarray
        predicted values; typically the output of a model
    yhat : ndarray or scalar
        truth or target values
    mask : ndarray, optional
        True where M should contribute to the cost, False where it should not

    Returns
    -------
    float, ndarray
        cost, dcost/dy

    """
    sub1 = 1 - y
    sub2 = 1 - yhat
    prefix = 1 / y.size
    cost = -prefix * (yhat * np.log(y) + sub2 * np.log(sub1)).sum()

    dcost = (-yhat / y) + (sub2 / sub1)
    dcost *= prefix
    return cost, dcost
