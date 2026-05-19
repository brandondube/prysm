"""Polynomial indexing tables shared by io_zemax and io_codev.

Thin re-exports of the index mappers already provided by
prysm.polynomials, so the IO modules import from a single private
surface:

- noll_to_nm: Zemax Standard / Noll Zernike index -> (n, m)
- fringe_to_nm: Code V Fringe Zernike index -> (n, m)
- xy_j_to_mn: XY-polynomial term index -> (m, n) where the term is
  x^m * y^n.  Matches both Zemax XYPOLY and Code V XYP conventions
  (piston at j=1, then x then y, then x^2 / xy / y^2, ...).

"""

from prysm.polynomials import noll_to_nm, fringe_to_nm, xy_j_to_mn  # noqa: F401
