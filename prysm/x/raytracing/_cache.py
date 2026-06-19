"""Cache helpers for raytracing derived quantities."""

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
