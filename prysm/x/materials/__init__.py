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
from .rii import RefractiveIndexCatalog, RefractiveIndexMaterial, default_db_path
from .fitted import FitReport, FittedMaterial, fit_material, from_samples
from . import lookup as _lookup

MIRROR = _lookup.MIRROR
air = _lookup.air
vacuum = _lookup.vacuum
glass = _lookup.glass
lookup = _lookup.lookup

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
    'RefractiveIndexCatalog',
    'RefractiveIndexMaterial',
    'StressOpticMaterial',
    'TabulatedMaterial',
    'TemperatureGridMaterial',
    'TemperatureSellmeierMaterial',
    'TemperatureShiftedMaterial',
    'ThicknessDependentMaterial',
    'air',
    'default_db_path',
    'fit_material',
    'from_samples',
    'glass',
    'load_agf_catalog',
    'lookup',
    'vacuum',
]
