"""Guard: nothing inside prysm/ may import numpy/scipy directly.

The backend shims in prysm.mathops are the only sanctioned route; direct
imports of `numpy` / `scipy` break the cupy / torch / mkl_fft swap path.

A small allow-list documents the modules where a real-numpy / direct-scipy
import is intentional (and the reason). Each entry below is the relative
path under prysm/, and the allowed imports for that file.

If a new direct import is genuinely needed, add it to ALLOWED with a
one-line justification; that justification is also expected to appear as a
top-of-file comment in the file itself.
"""
import ast
from pathlib import Path

import pytest

PRYSM_ROOT = Path(__file__).resolve().parent.parent / 'prysm'

# (file relative to prysm/, allowed direct-import modules)
ALLOWED = {
    # mathops itself is the shim; it MUST import numpy / scipy.
    'mathops.py': {'numpy', 'scipy', 'scipy.ndimage', 'scipy.interpolate',
                   'scipy.special', 'scipy.fft', 'scipy.optimize', 'scipy.signal'},
    # plotting wants matplotlib-compatible real-numpy arrays.
    'plotting.py': {'numpy'},
    'x/raytracing/plotting.py': {'numpy'},
    # scipy.spatial.Delaunay has no cupy/torch equivalent; geometry.py is
    # host-only by design.
    'geometry.py': {'scipy'},  # `from scipy import spatial`
    # F77 LBFGSB wraps a Fortran extension; it interacts with C and cannot
    # round-trip through a backend shim.
    'x/optym/_lbfgsb.py': {'numpy', 'scipy'},
}

# Files where `import numpy as truenp` is allowed; reason must appear as a
# top-of-file comment in the file itself.
TRUENP_ALLOWED = {
    'coordinates.py',
    'fttools.py',
    'geometry.py',
    'io.py',
    'segmented.py',
    'polynomials/zernike.py',
    'polynomials/xy.py',
    'x/fibers.py',
    'x/psi.py',
    'x/optym/linesearch.py',
}


def _iter_py_files():
    for path in PRYSM_ROOT.rglob('*.py'):
        yield path


def _collect_imports(tree):
    """Return list of (module_name, alias) for every import statement.

    `from scipy import spatial` and `from scipy.linalg.lapack import HAS_ILP64`
    both bind names within an already-imported module; we report the source
    module path (the `from X` part), letting the allow-list match by prefix.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, alias.asname))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.level:
                continue
            out.append((node.module, None))
    return out


def _is_allowed(module, allowed):
    """allow-list entries match by exact name or as a dotted-path prefix."""
    for entry in allowed:
        if module == entry or module.startswith(entry + '.'):
            return True
    return False


@pytest.mark.parametrize('path', list(_iter_py_files()), ids=lambda p: str(p.relative_to(PRYSM_ROOT)))
def test_no_direct_numpy_scipy(path):
    rel = path.relative_to(PRYSM_ROOT).as_posix()
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    imports = _collect_imports(tree)

    allowed = ALLOWED.get(rel, set())
    violations = []
    for module, asname in imports:
        top = module.split('.')[0]
        if top not in ('numpy', 'scipy'):
            continue
        # numpy as truenp is permitted on the truenp allow-list.
        if module == 'numpy' and asname == 'truenp' and rel in TRUENP_ALLOWED:
            continue
        if _is_allowed(module, allowed):
            continue
        violations.append(module)

    assert not violations, (
        f'{rel} directly imports {violations}; route through prysm.mathops '
        f'(or add to the allow-list in tests/test_no_direct_numpy_scipy.py '
        f'with a justification).'
    )
