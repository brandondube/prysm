"""Code V .seq (sequential lens) prescription reader.

Parses a sequential .seq into a list of Surface objects plus header
metadata, mirroring the PrescriptionFile contract of io_zemax.

Supported subset (raise informative error otherwise):
- Header: TITLE, DIM, WL, REF, EPD, YAN, XAN, RDM/CUM
- Surface boundaries: SO (object), S (new surface), SI (image), GO (end)
- Per-surface: RDY (radius Y), CUY (curvature Y), THI (thickness),
  CCY (conic Y), GLA (glass), ASP and A/B/C/D (even-asphere coefs)

Out of scope:
- Biconic (RDX/RDY pair with CUX/CCX)
- Toroid (TOR command)
- Zernike / XY polynomial surfaces (ZFR, XYP)
- Coordinate decenters (DECNTR, BEN)
- Solves, pickups, zoom configurations
- Anything inside DAR / @ blocks (macros, slip groups)

Commands are case-insensitive.  Per Code V, semicolons separate multiple
commands on the same physical line; commands may span lines via `&` (not
supported — newline ends the command).  Comment lines start with `!` and
are stripped.

"""

from .surfaces import PlaneSag, Surface
from . import materials as _materials
from ._indexing import fringe_to_nm, xy_j_to_mn
from ._io_common import fields_from_xy, read_text_or_path
from ._prescription import PrescriptionFile
from ._surface_spec import SurfaceSpec, build_surface


# ---------- tokenizer -------------------------------------------------------

def _strip_comment(line):
    """Drop '!'-prefixed comments and trailing whitespace."""
    i = line.find('!')
    if i >= 0:
        line = line[:i]
    return line.rstrip()


def _split_commands(text):
    """Split .seq text into a flat list of command-token lists.

    Each command is a list [verb, *args] of strings.  Commands are
    separated by newlines and by semicolons within a line.

    """
    cmds = []
    for line in text.splitlines():
        line = _strip_comment(line)
        if not line.strip():
            continue
        # split at semicolons
        for piece in line.split(';'):
            piece = piece.strip()
            if not piece:
                continue
            tokens = piece.split()
            tokens[0] = tokens[0].upper()
            cmds.append(tokens)
    return cmds


from ._io_common import parse_float as _parse_float  # noqa: E402  (kept name for callers)


# ---------- parser ----------------------------------------------------------

_KIND_BY_FNUM = {
    # Code V doesn't have a uniform 'field type' enum like Zemax FTYP;
    # YAN means Y angle (degrees), YIM means Y image height.  We map
    # whichever the user supplied.
}


def _new_surface_dict():
    return {
        'rdy': None,  # radius of curvature Y (mm)
        'cuy': None,  # curvature Y (1/mm) - alternative to RDY
        'rdx': None,  # radius X (biconic)
        'cux': None,
        'thi': 0.0,
        'ccy': 0.0,
        'ccx': None,  # set only for biconic
        'gla': None,
        'asphere_coefs': {},  # int order index -> coefficient value
        'is_asphere': False,
        'zfr_coefs': None,    # Fringe Zernike coefficient list
        'xyp_coefs': None,    # XY-polynomial coefficient list
        'nrr': None,          # normalization radius for Zernike / XY
        # decenter / tilt perturbation (Code V DEC/ADE/BDE/CDE)
        'dec_x': 0.0,
        'dec_y': 0.0,
        'ade': 0.0,  # alpha (tilt about X), degrees
        'bde': 0.0,  # beta (tilt about Y), degrees
        'cde': 0.0,  # gamma (tilt about Z), degrees
    }


def read_seq(path_or_text, *, _is_text=False, database=None):
    """Read a Code V .seq file into a PrescriptionFile.

    Parameters
    ----------
    path_or_text : str
    _is_text : bool
    database : refractivesqlite.Database, optional
        Required when any surface uses a real glass name. Air, blank, and
        mirror surfaces do not need a database.

    Returns
    -------
    PrescriptionFile

    """
    text, path_for_meta = read_text_or_path(path_or_text, is_text=_is_text)

    cmds = _split_commands(text)

    header = {
        'title': None,
        'unit': None,
        'wavelengths': [],
        'wavelength_weights': [],
        'reference_wvl_index': None,
        'epd': None,
        'yan': [],
        'xan': [],
        'extras': {},
    }
    radius_mode = True  # default: RDM (radius mode); CUM flips it
    surfaces = []           # list of surface dicts
    current = None          # currently-being-built surface dict
    stop_surface_idx = None  # 1-based ordinal among real surfaces

    def _commit_current():
        nonlocal current
        if current is not None:
            surfaces.append(current)
            current = None

    i = 0
    while i < len(cmds):
        verb, *args = cmds[i]
        if verb == 'LEN':
            pass  # header start marker
        elif verb in ('TITLE', 'TIT'):
            header['title'] = ' '.join(args)
        elif verb in ('RDM',):
            radius_mode = True
        elif verb in ('CUM',):
            radius_mode = False
        elif verb in ('DIM',):
            if args:
                u = args[0].upper()
                header['unit'] = {
                    'M': 'mm',  # CodeV 'M' = millimeter (confusingly)
                    'CM': 'cm', 'IN': 'in', 'FT': 'ft',
                }.get(u, u.lower())
        elif verb == 'WL':
            try:
                header['wavelengths'] = [float(t) for t in args]
            except ValueError:
                pass
        elif verb == 'WTW':
            try:
                header['wavelength_weights'] = [float(t) for t in args]
            except ValueError:
                pass
        elif verb == 'REF':
            if args:
                try:
                    header['reference_wvl_index'] = int(args[0])
                except ValueError:
                    pass
        elif verb == 'EPD':
            if args:
                try:
                    header['epd'] = float(args[0])
                except ValueError:
                    pass
        elif verb == 'YAN':
            try:
                header['yan'] = [float(t) for t in args]
            except ValueError:
                pass
        elif verb == 'XAN':
            try:
                header['xan'] = [float(t) for t in args]
            except ValueError:
                pass
        elif verb == 'STO':
            # 1-based ordinal of the stop surface (counted among S
            # entries between SO and SI)
            stop_surface_idx = len(surfaces) + (1 if current is not None else 1)
        elif verb == 'SO':
            # object surface; flush current, start a fresh surface dict
            # for object.  Anything inline after SO (e.g. ;THI 1E10)
            # comes as a separate command.  We use a placeholder dict.
            _commit_current()
            current = _new_surface_dict()
            current['_is_object'] = True
        elif verb == 'S':
            # new surface.  Inline tokens after S may be:
            # - 'RDY <r> THI <t> [GLA <g>]' positional shorthand
            # - bare radius / thickness for the shortest form
            _commit_current()
            current = _new_surface_dict()
            _consume_inline(args, current, radius_mode)
        elif verb == 'SI':
            _commit_current()
            current = _new_surface_dict()
            current['_is_image'] = True
        elif verb == 'GO':
            _commit_current()
            break
        # per-surface directives (apply to the most-recently-opened
        # surface)
        elif current is not None and verb == 'RDY':
            current['rdy'] = _parse_float(args[0])
        elif current is not None and verb == 'CUY':
            current['cuy'] = _parse_float(args[0])
        elif current is not None and verb == 'RDX':
            current['rdx'] = _parse_float(args[0])
        elif current is not None and verb == 'CUX':
            current['cux'] = _parse_float(args[0])
        elif current is not None and verb == 'THI':
            current['thi'] = _parse_float(args[0])
        elif current is not None and verb == 'CCY':
            current['ccy'] = _parse_float(args[0])
        elif current is not None and verb == 'CCX':
            current['ccx'] = _parse_float(args[0])
        elif current is not None and verb == 'GLA':
            current['gla'] = args[0] if args else None
        elif current is not None and verb == 'ASP':
            current['is_asphere'] = True
        elif current is not None and verb == 'ZFR':
            try:
                current['zfr_coefs'] = [_parse_float(t) for t in args]
            except ValueError:
                pass
        elif current is not None and verb == 'XYP':
            try:
                current['xyp_coefs'] = [_parse_float(t) for t in args]
            except ValueError:
                pass
        elif current is not None and verb in ('NRR', 'NRD'):
            # NRR: normalization radius for Zernike/XY; NRD: alternate spelling
            if args:
                try:
                    current['nrr'] = _parse_float(args[0])
                except ValueError:
                    pass
        elif current is not None and verb in ('DEC', 'DECNTR'):
            # DEC <x> <y> [<z>]; or DECNTR may be followed by axis-specific
            # tokens.  We support the (x, y, [z]) positional form most
            # commonly seen.
            if len(args) >= 1:
                current['dec_x'] = _parse_float(args[0])
            if len(args) >= 2:
                current['dec_y'] = _parse_float(args[1])
        elif current is not None and verb == 'ADE':
            if args:
                current['ade'] = _parse_float(args[0])
        elif current is not None and verb == 'BDE':
            if args:
                current['bde'] = _parse_float(args[0])
        elif current is not None and verb == 'CDE':
            if args:
                current['cde'] = _parse_float(args[0])
        elif current is not None and verb == 'BEN':
            # Code V "bend" -- coordinate-axis flip after a mirror.  prysm
            # handles reflection direction natively, so we silently ignore.
            pass
        elif current is not None and len(verb) == 1 and verb in 'ABCDEFGH':
            # A/B/C/D/... are even-asphere coefs (A = a4, B = a6, ...)
            order = ord(verb) - ord('A') + 1  # 1=A=a4, 2=B=a6, ...
            try:
                current['asphere_coefs'][order] = _parse_float(args[0])
                current['is_asphere'] = True
            except (IndexError, ValueError):
                pass
        else:
            header['extras'].setdefault(verb, []).append(' '.join(args))
        i += 1

    _commit_current()

    if not surfaces:
        raise ValueError('no surfaces found in .seq text')

    # Walk vertex z, skipping OBJECT THI (so first real surface lands at z=0)
    z = 0.0
    vertex_z = []
    for sd in surfaces:
        vertex_z.append(z)
        thi = sd.get('thi', 0.0)
        if sd.get('_is_object'):
            # object thickness establishes object distance; we skip from
            # the running z (object is upstream of surface 1 at z=0)
            continue
        z += thi

    out_surfaces = []
    out_origin = []
    for idx, sd in enumerate(surfaces):
        if sd.get('_is_object'):
            continue
        if sd.get('_is_image'):
            # image plane (typ='eval')
            img = Surface(shape=PlaneSag(), typ='eval',
                          P=[0.0, 0.0, vertex_z[idx]])
            out_surfaces.append(img)
            out_origin.append(idx)
            continue
        s = _build_surface(sd, vertex_z[idx], radius_mode, database)
        out_surfaces.append(s)
        out_origin.append(idx)

    # translate STO from 1-based-among-real-surfaces to our index
    stop_index = None
    if stop_surface_idx is not None:
        # count physical surfaces (skip OBJECT and IMAGE flags)
        # the STO command was recorded as 1-based on the running real
        # surface count; map back via out_origin
        real_counter = 0
        target = stop_surface_idx
        for k, origin_idx in enumerate(out_origin):
            sd = surfaces[origin_idx]
            if sd.get('_is_object') or sd.get('_is_image'):
                continue
            real_counter += 1
            if real_counter == target:
                stop_index = k
                break

    # Field objects from YAN / XAN
    fields = fields_from_xy(header['xan'], header['yan'],
                            kind='angle', unit='deg')

    return PrescriptionFile(
        surfaces=out_surfaces,
        wavelengths=header['wavelengths'],
        epd=header['epd'],
        stop_index=stop_index,
        fields=fields,
        unit=header['unit'],
        source_path=path_for_meta,
        source_format='codev',
        extras=header['extras'],
    )


def _consume_inline(args, sd, radius_mode):
    """Pull (RDY|CUY|THI|GLA|CCY) <value> token pairs from the inline
    args of an 'S' command."""
    i = 0
    while i < len(args):
        tok = args[i].upper()
        if tok == 'RDY' and i + 1 < len(args):
            sd['rdy'] = _parse_float(args[i + 1])
            i += 2
        elif tok == 'CUY' and i + 1 < len(args):
            sd['cuy'] = _parse_float(args[i + 1])
            i += 2
        elif tok == 'THI' and i + 1 < len(args):
            sd['thi'] = _parse_float(args[i + 1])
            i += 2
        elif tok == 'CCY' and i + 1 < len(args):
            sd['ccy'] = _parse_float(args[i + 1])
            i += 2
        elif tok == 'GLA' and i + 1 < len(args):
            sd['gla'] = args[i + 1]
            i += 2
        else:
            i += 1  # unknown inline token; skip silently


def _build_surface(sd, vertex_z, radius_mode, database=None):
    """Turn one parsed Code V surface dict into a prysm Surface."""
    c_y = _resolve_c(sd, 'cuy', 'rdy')
    c_x = _resolve_c(sd, 'cux', 'rdx')
    k_y = float(sd.get('ccy', 0.0))
    k_x = sd.get('ccx', None)

    glass = sd.get('gla')
    if glass is not None and glass.upper() in ('REFL', 'REF_S', 'REFL_FRONT'):
        n_callable = _materials.MIRROR
    else:
        n_callable = _materials.lookup(glass, database=database)
    is_mirror = (n_callable is _materials.MIRROR)
    typ = 'refl' if is_mirror else 'refr'
    n_arg = None if is_mirror else n_callable
    P = [0.0, 0.0, float(vertex_z)]

    # tilt / decenter kwargs (Code V uses degrees by default for ADE/BDE/CDE)
    tilt = None
    decenter = None
    if any(sd.get(k, 0.0) != 0.0 for k in ('ade', 'bde', 'cde')):
        # prysm tilt convention is (rz, ry, rx).  Code V: ADE = tilt
        # about X axis (alpha), BDE = about Y (beta), CDE = about Z
        # (gamma).
        tilt = (float(sd.get('cde', 0.0)),
                float(sd.get('bde', 0.0)),
                float(sd.get('ade', 0.0)))
    if sd.get('dec_x', 0.0) != 0.0 or sd.get('dec_y', 0.0) != 0.0:
        decenter = (float(sd['dec_x']), float(sd['dec_y']), 0.0)

    common = dict(tilt=tilt, decenter=decenter)

    # Zernike (Fringe) surface
    if sd.get('zfr_coefs') is not None:
        coefs = sd['zfr_coefs']
        nrr = sd.get('nrr') or 1.0
        nms = [fringe_to_nm(j) for j in range(1, len(coefs) + 1)]
        return build_surface(SurfaceSpec(
            'zernike', typ, P, n_arg,
            dict(c=c_y, k=k_y, normalization_radius=float(nrr),
                 nms=nms, coefs=tuple(coefs), norm=False),
            **common))

    # XY polynomial surface
    if sd.get('xyp_coefs') is not None:
        coefs = sd['xyp_coefs']
        nrr = sd.get('nrr') or 1.0
        mns = [xy_j_to_mn(j) for j in range(1, len(coefs) + 1)]
        return build_surface(SurfaceSpec(
            'xy', typ, P, n_arg,
            dict(c=c_y, k=k_y, normalization_radius=float(nrr),
                 mns=mns, coefs=tuple(coefs)),
            **common))

    # Biconic (anisotropic curvature on the two axes)
    if c_x is not None or k_x is not None:
        cx_resolved = c_x if c_x is not None else c_y
        kx_resolved = float(k_x) if k_x is not None else 0.0
        return build_surface(SurfaceSpec(
            'biconic', typ, P, n_arg,
            dict(c_x=cx_resolved, c_y=c_y, k_x=kx_resolved, k_y=k_y),
            **common))

    if sd.get('is_asphere'):
        coefs_dict = sd.get('asphere_coefs', {})
        if coefs_dict:
            n_coefs = max(coefs_dict)
            coefs = tuple(coefs_dict.get(i, 0.0)
                          for i in range(1, n_coefs + 1))
        else:
            coefs = ()
        return build_surface(SurfaceSpec(
            'even_asphere', typ, P, n_arg,
            dict(c=c_y, k=k_y, coefs=coefs), **common))

    return build_surface(SurfaceSpec('conic', typ, P, n_arg,
                                     dict(c=c_y, k=k_y), **common))


def _resolve_c(sd, cu_key, rd_key):
    """Return the curvature stored under cu_key, or 1/rd if rd_key is set,
    else None when neither is present (so callers can detect biconic
    presence via 'c_x is None')."""
    if sd.get(cu_key) is not None:
        return float(sd[cu_key])
    if sd.get(rd_key) is not None:
        r = float(sd[rd_key])
        if _finite_nonzero(r):
            return 1.0 / r
        return 0.0
    if cu_key == 'cuy':
        # default along Y: no curvature
        return 0.0
    # biconic X-axis defaults: signal "not set"
    return None


def _finite_nonzero(x):
    from math import isfinite
    return isfinite(x) and x != 0.0
