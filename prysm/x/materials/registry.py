"""Searchable material registry with computed-property filters."""

from .catalog import (
    CatalogChain,
    RecordSet,
    _record_matches_filters,
    _record_matches_query,
)
from .core import MissingKError


_COMPUTED_CRITERIA = {
    'n_at',
    'k_max',
    'dispersion',
    'partial_dispersion',
    'abbe',
}


class MaterialRegistry(RecordSet):
    """Index many catalogs and search metadata or computed properties.

    A RecordSet (so material_for_name and namespaced lookup come for free); the
    overridden search adds computed-property criteria on top of metadata search.
    """

    def __init__(self, records):
        self._records = tuple(records)
        self._metric_cache = {}

    @classmethod
    def from_catalogs(cls, catalogs):
        """Build a registry from a catalog, chain, or iterable of catalogs."""
        if isinstance(catalogs, CatalogChain):
            return cls(tuple(catalogs.records()))
        if hasattr(catalogs, 'records'):
            return cls(tuple(catalogs.records()))
        records = []
        for catalog in catalogs:
            records.extend(tuple(catalog.records()))
        return cls(records)

    def records(self):
        """Return registry records."""
        return self._records

    def search(self, **criteria):
        """Return records matching metadata and computed filters."""
        return list(self.iter_search(**criteria))

    def iter_search(self, **criteria):
        """Yield records matching metadata and computed filters."""
        for record in self._records:
            if self._matches(record, criteria):
                yield record

    def _matches(self, record, criteria):
        query = criteria.get('query')
        metadata_filters = {
            key: value for key, value in criteria.items()
            if key != 'query' and key not in _COMPUTED_CRITERIA
        }
        if not _record_matches_query(record, query):
            return False
        if not _record_matches_filters(record, metadata_filters):
            return False
        if 'n_at' in criteria and criteria['n_at'] is not None:
            wvl, lo, hi, temperature = _criterion_tuple(
                'n_at', criteria['n_at'], 3, 4, None
            )
            n = self._metric(record, 'n_at', (wvl, temperature))
            if (lo is not None and n < lo) or (hi is not None and n > hi):
                return False
        if 'k_max' in criteria and criteria['k_max'] is not None:
            wvl, threshold, temperature = _criterion_tuple(
                'k_max', criteria['k_max'], 2, 3, None
            )
            if threshold is None:
                raise ValueError('k_max criterion requires a non-None threshold')
            k = self._metric(record, 'k_at', (wvl, temperature))
            if k > threshold:
                return False
        if 'dispersion' in criteria and criteria['dispersion'] is not None:
            wvl1, wvl2, lo, hi, temperature = _criterion_tuple(
                'dispersion', criteria['dispersion'], 4, 5, None
            )
            value = self._metric(record, 'dispersion', (wvl1, wvl2, temperature))
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                return False
        if 'partial_dispersion' in criteria and criteria['partial_dispersion'] is not None:
            w1, w2, w3, w4, lo, hi, temperature = _criterion_tuple(
                'partial_dispersion',
                criteria['partial_dispersion'],
                6,
                7,
                None,
            )
            value = self._metric(
                record,
                'partial_dispersion',
                (w1, w2, w3, w4, temperature),
            )
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                return False
        if 'abbe' in criteria and criteria['abbe'] is not None:
            ws, wc, wl, lo, hi, temperature = _criterion_tuple(
                'abbe', criteria['abbe'], 5, 6, None
            )
            value = self._metric(record, 'abbe', (ws, wc, wl, temperature))
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                return False
        return True

    def _metric(self, record, metric, args):
        key = (record.material_id, metric, args)
        try:
            if key in self._metric_cache:
                return self._metric_cache[key]
        except TypeError:
            # array-valued criterion args are unhashable; skip the cache.
            key = None
        material = record.load()
        if metric == 'n_at':
            wvl, temperature = args
            value = material.n_at(wvl, temperature=temperature)
        elif metric == 'k_at':
            wvl, temperature = args
            try:
                value = material.k(wvl, temperature=temperature)
            except MissingKError:
                # no extinction data -> treat as transparent for the k_max filter.
                value = 0.0
        elif metric == 'dispersion':
            wvl1, wvl2, temperature = args
            value = material.dispersion(wvl1, wvl2, temperature=temperature)
        elif metric == 'partial_dispersion':
            w1, w2, w3, w4, temperature = args
            value = material.partial_dispersion(w1, w2, w3, w4, temperature=temperature)
        elif metric == 'abbe':
            ws, wc, wl, temperature = args
            value = material.abbe(ws, wc, wl, temperature=temperature)
        else:
            raise ValueError(f'unknown metric {metric!r}')
        if key is not None:
            self._metric_cache[key] = value
        return value


def _criterion_tuple(name, value, min_length, max_length, fill):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f'{name} criterion must be a sequence') from exc
    if len(values) < min_length or len(values) > max_length:
        raise ValueError(
            f'{name} criterion expects {min_length} to {max_length} values'
        )
    return values + (fill,) * (max_length - len(values))
