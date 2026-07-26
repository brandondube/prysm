"""Cache helpers for raytracing derived quantities."""

import numbers

from prysm.mathops import np, array_to_true_numpy

_MISS = object()


class StateCache(dict):
    """dict with get_or_compute and cached-None support."""

    __slots__ = ()

    def get_or_compute(self, key, compute):
        value = self.get(key, _MISS)
        if value is _MISS:
            value = compute()
            self[key] = value
        return value


def structural_key(value):
    """Stable, hashable key for nested public analysis arguments."""
    if value is None or isinstance(value, (str, bytes, bool, numbers.Number)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((k, structural_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(structural_key(v) for v in value)
    kind = getattr(value, 'kind', None)
    opts = getattr(value, 'opts', None)
    if kind is not None and opts is not None:
        return ('Sampling', kind, structural_key(opts))
    if all(hasattr(value, name) for name in ('hx', 'hy', 'kind', 'unit')):
        return (
            'Field', value.hx, value.hy, value.kind, value.unit,
            getattr(value, 'object_z', None),
            structural_key(getattr(value, 'vignetting', None)),
        )
    try:
        concrete = array_to_true_numpy(np.asarray(value))
        return ('array', tuple(concrete.shape), str(concrete.dtype),
                tuple(concrete.ravel().tolist()))
    except (TypeError, ValueError):
        raise TypeError(
            f'cannot construct a structural cache key for {type(value).__name__}'
        ) from None


__all__ = ['StateCache', 'structural_key']
