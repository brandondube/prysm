"""prysm, a python optics module."""
from pkg_resources import get_distribution

# revisit the decision to export anything at the top level or not
__version__ = get_distribution('prysm').version
