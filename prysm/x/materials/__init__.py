"""Experimental optical materials and catalog package."""

from .core import (
    BaseMaterial,
    ConstantMaterial,
    FormulaMaterial,
    MaterialProtocol,
    MaterialRecord,
    MaterialRangeError,
    MissingKError,
)
from .tabulated import MaterialData, TabulatedMaterial, TemperatureGridMaterial
from .charms import (
    CHARMSCoefficientMaterial,
    CHARMSDataset,
    CHARMSTableMaterial,
    TemperatureSellmeierMaterial,
)
from .catalog import AmbiguousMaterialError, Catalog, CatalogChain
from .registry import MaterialRegistry
from .transforms import (
    IndexOffsetMaterial,
    MaterialTransform,
    ProcessVariantMaterial,
    StressOpticMaterial,
    TemperatureShiftedMaterial,
    ThicknessDependentMaterial,
)
from .agf import AGFCatalog, AGFMaterial, load_agf_catalog
from .rii import RefractiveIndexCatalog, RefractiveIndexMaterial, default_cache_root
from .fitted import FitReport, FittedMaterial, fit_material, from_samples
from . import lookup as _lookup

Database = _lookup.Database
MIRROR = _lookup.MIRROR
RefractiveIndexDatabase = _lookup.RefractiveIndexDatabase
SQLiteMaterial = _lookup.SQLiteMaterial
air = _lookup.air
vacuum = _lookup.vacuum


def lookup(name, database=None):
    """Resolve a glass token to a callable material, air, or MIRROR."""
    return _lookup.lookup(name, database=database, database_type=Database)


def glass(name, database=None):
    """Resolve a glass name from a material catalog or refractivesqlite database."""
    return _lookup.glass(name, database=database, database_type=Database)


def load_material_db():
    """Load the refractiveindex.info database from the prysm repo root."""
    return _lookup.load_material_db(database_type=Database)

__all__ = [
    'AGFCatalog',
    'AGFMaterial',
    'AmbiguousMaterialError',
    'BaseMaterial',
    'Catalog',
    'CatalogChain',
    'CHARMSCoefficientMaterial',
    'CHARMSDataset',
    'CHARMSTableMaterial',
    'ConstantMaterial',
    'Database',
    'FitReport',
    'FittedMaterial',
    'FormulaMaterial',
    'IndexOffsetMaterial',
    'MIRROR',
    'MaterialData',
    'MaterialProtocol',
    'MaterialRecord',
    'MaterialRangeError',
    'MaterialRegistry',
    'MaterialTransform',
    'MissingKError',
    'ProcessVariantMaterial',
    'RefractiveIndexDatabase',
    'RefractiveIndexCatalog',
    'RefractiveIndexMaterial',
    'SQLiteMaterial',
    'StressOpticMaterial',
    'TabulatedMaterial',
    'TemperatureGridMaterial',
    'TemperatureSellmeierMaterial',
    'TemperatureShiftedMaterial',
    'ThicknessDependentMaterial',
    'air',
    'default_cache_root',
    'fit_material',
    'from_samples',
    'glass',
    'load_agf_catalog',
    'load_material_db',
    'lookup',
    'vacuum',
]
