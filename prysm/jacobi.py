"""High performance / recursive jacobi polynomial calculation."""
from .mathops import engine as e


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    one_minus_x = 1 - x
    return (one_minus_x ** alpha) * (one_minus_x ** beta)


def a(n, alpha, beta, x):
    """The leading term of the recurrence relation from Wikipedia, * P_n^(a,b)."""
    term1 = 2 * n
    term2 = n + alpha + beta
    term3 = 2 * n + alpha + beta - 2
    return term1 * term2 * term3


def b(n, alpha, beta, x):
    """The second term of the recurrence relation from Wikipedia, * P_n-1^(a,b)."""
    term1 = 2 * n + alpha + beta - 1
    iterm1 = 2 * n + alpha + beta
    iterm2 = 2 * n + alpha + beta - 2
    iterm3 = alpha ** 2 - beta ** 2
    temp_product = iterm1 * iterm2 * x + iterm3
    return term1 * temp_product


def c(n, alpha, beta, x):
    """The third term of the recurrence relation from Wikipedia, * P_n-2^(a,b)."""
    term1 = 2 * (n + alpha - 1)
    term2 = (n + beta - 1)
    term3 = (2 * n + alpha + beta)
    return term1 * term2 * term3


def jacobi(n, alpha, beta, x, Pnm1=None, Pnm2=None):
    if n == 0:
        return e.ones_like(x)
    elif n == 1:
        term1 = alpha + 1
        term2 = alpha + beta + 2
        term3 = (x - 1) / 2
        return term1 + term2 * term3
    elif n > 1:
        if Pnm1 is None:
            Pnm1 = jacobi(n-1, alpha, beta, x)
        if Pnm2 is None:
            Pnm2 = jacobi(n-2, alpha, beta, x)

        a_ = a(n, alpha, beta, x)
        b_ = b(n, alpha, beta, x)
        c_ = c(n, alpha, beta, x)
        term1 = b_ * Pnm1
        term2 = c_ * Pnm2
        tmp = term1 - term2
        return tmp / a_
