"""Experimental optical materials and catalog package."""

from .core import (
    BaseMaterial,
    ConstantMaterial,
    FormulaMaterial,
    MaterialProtocol,
    MaterialRecord,
    MaterialRangeError,
    MissingKError,
    model_glass,
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
    IsothermalMaterial,
    MaterialTransform,
    ProcessVariantMaterial,
    StressOpticMaterial,
    TemperatureShiftedMaterial,
    ThicknessDependentMaterial,
)
from .infrared import (
    charms_germanium,
    charms_silicon,
    sapphire_ordinary,
    infrared_catalog,
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
resolve_index = _lookup.resolve_index

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
    'IsothermalMaterial',
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
    'charms_germanium',
    'charms_silicon',
    'default_db_path',
    'fit_material',
    'infrared_catalog',
    'from_samples',
    'glass',
    'load_agf_catalog',
    'lookup',
    'model_glass',
    'resolve_index',
    'sapphire_ordinary',
    'vacuum',
]
