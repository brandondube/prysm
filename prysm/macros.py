"""Useful macros for performing more complicated tasks."""
from collections import namedtuple

from .seidel import Seidel
from .fringezernike import FringeZernike
from .otf import MTF
from .mtf_utils import mtf_ts_extractor, mtf_ts_to_dataframe
from .thinlens import defocus_to_image_displacement

from prysm import mathops as m

SystemConfig = namedtuple('SystemConfig', ('efl', 'fno', 'wvl', 'samples', 'mask', 'mask_target'))
SimulationConfig = namedtuple('SimulationConfig',
                              ('freqs',
                               'focus_range_waves',
                               'focus_zernike',
                               'focus_normed',
                               'focus_planes',
                               ) + SystemConfig._fields)
SystemConfig.__new__.__defaults__ = ('circle', 'fcn')
SimulationConfig.__new__.__defaults__ = ('circle', 'fcn')

DEFAULT_SIM_PARAMS = SimulationConfig(
    efl=50,
    fno=2,
    wvl=0.55,
    samples=128,
    freqs=tuple(range(10, 850, 10)),
    focus_range_waves=0.5 / m.sqrt(3),
    focus_zernike=True,
    focus_normed=True,
    focus_planes=21)


def thrufocus_mtf_from_wavefront(focused_wavefront, sim_params):
    """Create a thru-focus T/S MTF curve at each frequency requested from a focused wavefront.

    Parameters
    ----------
    focused_wavefront : `Pupil`
        a pupil object

    sim_params : `SimulationConfig`
        a SimulationConfig namedtuple

    Returns
    -------
    `pandas.DataFrame`
        dataframe of data

    Notes
    -----
    see macros.DEFAULT_SIM_PARAMS for an example config

    """
    import pandas as pd
    s = sim_params
    focusdiv_wvs = m.linspace(-1*s.focus_range_waves, s.focus_range_waves, s.focus_planes)
    focusdiv_um = defocus_to_image_displacement(focusdiv_wvs, s.fno, s.wvl, s.focus_zernike, s.focus_normed)
    dfs = []
    for focus, displacement in zip(focusdiv_wvs, focusdiv_um):
        if s.focus_zernike:
            defocus = FringeZernike(base=1, Z4=focus, rms_norm=s.focus_normed, samples=s.samples,
                                    epd=s.efl / s.fno,
                                    wavelength=s.wvl,
                                    mask=s.mask,
                                    mask_target=s.mask_target)
        else:
            defocus = Seidel(W020=focus,
                             epd=s.efl / s.fno,
                             samples=s.samples,
                             wavelength=s.wvl,
                             mask=s.mask,
                             mask_target=s.mask_target)
        mtf = MTF.from_pupil(focused_wavefront + defocus, efl=s.efl)
        tan, sag = mtf_ts_extractor(mtf, s.freqs)
        dfs.append(mtf_ts_to_dataframe(tan, sag, s.freqs, focus=displacement))
    return pd.concat(dfs)


def thrufocus_mtf_from_wavefront_array(focused_wavefront, sim_params):
    """Create a thru-focus T/S MTF curve at each frequency requested from a focused wavefront.

    TODO: refactor

    Parameters
    ----------
    focused_wavefront : `Pupil`
        a pupil object

    sim_params : `SimulationConfig`
        a SimulationConfig namedtuple

    Returns
    -------
    `pandas.DataFrame`
        dataframe of data

    Notes
    -----
    see marcros.DEFAULT_SIM_PARAMS for an example config.

    """
    s = sim_params
    focusdiv_wvs = m.linspace(-s.focus_range_waves, s.focus_range_waves, s.focus_planes)
    tt, ss = m.empty((s.focus_planes, len(s.freqs))), m.empty((s.focus_planes, len(s.freqs)))
    for idx, focus in enumerate(focusdiv_wvs):
        if s.focus_zernike:
            defocus = FringeZernike(base=1, Z4=focus, rms_norm=s.focus_normed, samples=s.samples,
                                    epd=s.efl / s.fno,
                                    wavelength=s.wvl)
        else:
            defocus = Seidel(W020=focus,
                             epd=s.efl / s.fno,
                             samples=s.samples,
                             wavelength=s.wvl)
        mtf = MTF.from_pupil(focused_wavefront + defocus, efl=s.efl)
        tan, sag = mtf_ts_extractor(mtf, s.freqs)
        tt[idx, :] = tan
        ss[idx, :] = sag

    return tt, ss
