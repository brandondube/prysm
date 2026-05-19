"""Compatibility facade for raytracing surface types and calculus."""

from . import _surfaces_impl as _impl

_NAMES = [name for name in dir(_impl) if not name.startswith('__')]

globals().update({name: getattr(_impl, name) for name in _NAMES})

__all__ = [name for name in _NAMES if not name.startswith('_')]
