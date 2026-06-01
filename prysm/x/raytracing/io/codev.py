"""Code V .seq (sequential lens) prescription reader.

Parses a sequential .seq into a LensData (sequential rows + metadata),
mirroring the contract of zemax.read_zmx.

Supported subset (raise informative error otherwise):
- Header: TITLE, DIM, WL, REF, EPD, YAN, XAN, RDM/CUM
- Surface boundaries: SO (object), S (new surface), SI (image), GO (end).
  Each carries Code V free-format positional data: radius (or curvature in
  CUM mode), thickness, then an optional glass name.  An older keyword
  spelling (RDY/CUY/THI/GLA value) is also accepted.
- Per-surface: RDY (radius Y), CUY (curvature Y), THI (thickness),
  K (conic Y), GLA (glass), ASP and A/B/C/D (even-asphere coefs)
- Decenter / tilt: XDE/YDE/ZDE (decenter), ADE/BDE/CDE (tilt, degrees),
  and the DAR (decenter-and-return) block flag

Out of scope:
- Toroid (TOR command)
- Solves, pickups, zoom configurations, vignetting (VUY/VLY)
- BEN/REV/RET decenter types, GLB/GLO global references

The C-suffixed codes (CCY, CCX, THC, KC, AC, ADC, BDC, CDC, XDC, ...) are Code V
optimization coupling/control codes, not geometry; they are ignored.  In
particular CCY is the Y-curvature coupling code -- the conic constant is K.

Code V wavelengths (WL) are in nanometers; prysm uses microns, so they are
divided by 1000 on import.

Commands are case-insensitive.  Per Code V, semicolons separate multiple
commands on the same physical line; commands may span lines via `&` (not
supported — newline ends the command).  Comment lines start with `!` and
are stripped.

"""

from ..surfaces import Plane
from .. import materials as _materials
from ._indexing import fringe_to_nm, xy_j_to_mn
from ._common import (
    fields_from_xy,
    read_text_or_path,
    fold_sign,
    writable_shape_or_raise,
)
from ..lensdata import LensData
from ._surface_spec import SurfaceSpec, build_shape


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


from ._common import parse_float as _parse_float  # noqa: E402  (kept name for callers)


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
        'rdx': None,  # radius X (toroid / anamorph)
        'cux': None,
        'thi': 0.0,
        'k': 0.0,     # conic constant Y (Code V K)
        'kx': None,   # conic constant X (set only for biconic / anamorph)
        'gla': None,
        'asphere_coefs': {},  # int order index -> coefficient value
        'is_asphere': False,
        'zfr_coefs': None,    # Fringe Zernike coefficient list
        'xyp_coefs': None,    # XY-polynomial coefficient list
        'nrr': None,          # normalization radius for Zernike / XY
        # decenter / tilt perturbation (Code V XDE/YDE/ZDE, ADE/BDE/CDE)
        'dec_x': 0.0,
        'dec_y': 0.0,
        'dec_z': 0.0,
        'ade': 0.0,  # alpha (tilt about X), degrees
        'bde': 0.0,  # beta (tilt about Y), degrees
        'cde': 0.0,  # gamma (tilt about Z), degrees
        'dar': False,  # decenter-and-return block (vs persistent basic break)
    }


def read_seq(path_or_text, *, _is_text=False, database=None):
    """Read a Code V .seq file into a LensData.

    Parameters
    ----------
    path_or_text : str
    _is_text : bool
    database : refractivesqlite.Database, optional
        Required when any surface uses a real glass name. Air, blank, and
        mirror surfaces do not need a database.

    Returns
    -------
    LensData

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
    stop_surface = None     # the surface dict that STO marks

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
            # Code V specifies wavelengths in nanometers; prysm uses microns.
            try:
                header['wavelengths'] = [float(t) / 1000.0 for t in args]
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
            # STO marks the surface whose block it appears in -- the still-open
            # surface, or the most recently committed one if none is open.  We
            # record the dict itself and resolve its compiled index when the
            # rows are built, which is robust whether or not an object (SO) was
            # written and regardless of intervening coordinate-break rows.
            stop_surface = current if current is not None else (
                surfaces[-1] if surfaces else None)
        elif verb == 'SO':
            # object surface.  Code V free format gives positional radius and
            # thickness (e.g. SO 0. 1E10); consume them so they are not lost.
            _commit_current()
            current = _new_surface_dict()
            current['_is_object'] = True
            _consume_surface_line(args, current, radius_mode)
        elif verb == 'S':
            # new surface.  Inline tokens after S are Code V free-format
            # positional radius / thickness / glass, or an older keyword form.
            _commit_current()
            current = _new_surface_dict()
            _consume_surface_line(args, current, radius_mode)
        elif verb == 'SI':
            _commit_current()
            current = _new_surface_dict()
            current['_is_image'] = True
            _consume_surface_line(args, current, radius_mode)
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
        elif current is not None and verb == 'K':
            current['k'] = _parse_float(args[0])
        elif current is not None and verb == 'KX':
            current['kx'] = _parse_float(args[0])
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
        elif current is not None and verb == 'DAR':
            # decenter-and-return block: the decenter/tilt that follow apply to
            # this surface only; the axis returns afterward.
            current['dar'] = True
        elif current is not None and verb == 'XDE':
            if args:
                current['dec_x'] = _parse_float(args[0])
        elif current is not None and verb == 'YDE':
            if args:
                current['dec_y'] = _parse_float(args[0])
        elif current is not None and verb == 'ZDE':
            if args:
                current['dec_z'] = _parse_float(args[0])
        elif current is not None and verb in ('DEC', 'DECNTR'):
            # legacy positional decenter DEC <x> <y>; real Code V uses XDE/YDE.
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

    # Field objects from YAN / XAN
    fields = fields_from_xy(header['xan'], header['yan'],
                            kind='angle', unit='deg')
    ref_idx = header.get('reference_wvl_index')
    wavelengths = header['wavelengths']
    reference_wavelength = None
    if ref_idx is not None and 1 <= ref_idx <= len(wavelengths):
        reference_wavelength = float(wavelengths[ref_idx - 1])

    ld = LensData(
        epd=header['epd'], fields=fields, wavelengths=wavelengths,
        reference_wavelength=reference_wavelength, unit=header['unit'],
        source_path=path_for_meta, source_format='codev',
        extras=header['extras'],
    )

    # Build rows.  The object surface only carries object-space thickness,
    # which lands the first real surface at z=0; we skip it.  Per-surface
    # Code V decenter/tilt (XDE/YDE/ZDE, ADE/BDE/CDE) becomes a coordinate
    # break -- decenter-and-return (DAR) when the surface declared a DAR block,
    # otherwise a persistent basic break.
    # Code V encodes post-mirror gaps as negative thicknesses on an unfolded
    # axis; LensData folds the frame at each reflection and keeps thickness
    # positive.  Convert by negating the gap once per preceding reflection.
    n_refl = 0
    compiled_idx = 0  # index of the next surface among compiled surfaces
    for sd in surfaces:
        if sd.get('_is_object'):
            continue
        tilt, decenter, kind = _pose_from_dict(sd)
        if tilt is not None or decenter is not None:
            ld.add_coordbreak(
                decenter=decenter or (0.0, 0.0, 0.0),
                tilt=tilt or (0.0, 0.0, 0.0), kind=kind)
        if sd.get('_is_image'):
            sign = fold_sign(n_refl)
            ld.add(Plane(), typ='eval',
                   thickness=sign * float(sd.get('thi', 0.0)))
        else:
            spec = _build_spec(sd, radius_mode, database)
            if spec.typ == 'refl':
                n_refl += 1
            sign = fold_sign(n_refl)
            ld.add(build_shape(spec),
                   thickness=sign * float(sd.get('thi', 0.0)),
                   material=spec.n, typ=spec.typ)
        if sd is stop_surface:
            ld.stop_index = compiled_idx
        compiled_idx += 1

    return ld


# keywords whose presence as the first inline token of an SO/S/SI line means
# the older keyword spelling is in use rather than the positional free format.
_SURFACE_KEYWORDS = ('RDY', 'CUY', 'RDX', 'CUX', 'THI', 'GLA', 'K')


def _is_number(token):
    """True if token parses as a Code V numeric (including INF / INFINITY)."""
    t = token.strip()
    if t.upper() in ('INF', 'INFINITY'):
        return True
    try:
        float(t)
        return True
    except ValueError:
        return False


def _consume_surface_line(args, sd, radius_mode):
    """Parse the inline tokens of an SO / S / SI command.

    Code V free format gives radius (or curvature in CUM mode), thickness, and
    an optional glass name positionally:  S <rad> <thi> [glass].  An older
    keyword spelling (RDY/CUY/THI/GLA value ...) is also accepted; the two are
    told apart by whether the first token is a recognized data keyword.

    """
    if args and args[0].upper() in _SURFACE_KEYWORDS:
        _consume_inline_keywords(args, sd)
        return
    # positional: <radius/curvature> [<thickness>] [<glass>]
    pos = 0
    if pos < len(args) and _is_number(args[pos]):
        val = _parse_float(args[pos])
        sd['rdy' if radius_mode else 'cuy'] = val
        pos += 1
    if pos < len(args) and _is_number(args[pos]):
        sd['thi'] = _parse_float(args[pos])
        pos += 1
    if pos < len(args):
        sd['gla'] = args[pos]


def _consume_inline_keywords(args, sd):
    """Pull (RDY|CUY|RDX|CUX|THI|GLA|K) <value> token pairs from inline args."""
    i = 0
    keys = {'RDY': 'rdy', 'CUY': 'cuy', 'RDX': 'rdx', 'CUX': 'cux',
            'THI': 'thi', 'K': 'k'}
    while i < len(args):
        tok = args[i].upper()
        if tok in keys and i + 1 < len(args):
            sd[keys[tok]] = _parse_float(args[i + 1])
            i += 2
        elif tok == 'GLA' and i + 1 < len(args):
            sd['gla'] = args[i + 1]
            i += 2
        else:
            i += 1  # unknown / control-code inline token; skip silently


def _pose_from_dict(sd):
    """Coordinate-break (tilt, decenter, kind) for one parsed surface dict.

    Code V uses degrees for ADE/BDE/CDE.  prysm tilt convention is (rz, ry, rx);
    Code V alpha/beta are left-handed, so invert ADE/BDE at this boundary only.
    A surface with a DAR block gets a decenter-and-return break; otherwise a
    persistent basic break.

    """
    tilt = None
    decenter = None
    if any(sd.get(k, 0.0) for k in ('ade', 'bde', 'cde')):
        tilt = (float(sd.get('cde', 0.0)),
                -float(sd.get('bde', 0.0)),
                -float(sd.get('ade', 0.0)))
    if any(sd.get(k, 0.0) for k in ('dec_x', 'dec_y', 'dec_z')):
        decenter = (float(sd.get('dec_x', 0.0)),
                    float(sd.get('dec_y', 0.0)),
                    float(sd.get('dec_z', 0.0)))
    kind = 'dar' if sd.get('dar') else 'basic'
    return tilt, decenter, kind


def _build_spec(sd, radius_mode, database=None):
    """Turn one parsed Code V surface dict into a SurfaceSpec (no pose)."""
    c_y = _resolve_c(sd, 'cuy', 'rdy')
    c_x = _resolve_c(sd, 'cux', 'rdx')
    k_y = float(sd.get('k', 0.0))
    k_x = sd.get('kx', None)

    glass = sd.get('gla')
    if glass is not None and glass.upper() in ('REFL', 'REF_S', 'REFL_FRONT'):
        n_callable = _materials.MIRROR
    else:
        n_callable = _lookup_codev_glass(glass, database)
    is_mirror = (n_callable is _materials.MIRROR)
    typ = 'refl' if is_mirror else 'refr'
    n_arg = None if is_mirror else n_callable

    def spec(kind, params):
        return SurfaceSpec(kind, typ, None, n_arg, params)

    # Zernike (Fringe) surface
    if sd.get('zfr_coefs') is not None:
        coefs = sd['zfr_coefs']
        nrr = sd.get('nrr') or 1.0
        nms = [fringe_to_nm(j) for j in range(1, len(coefs) + 1)]
        return spec('zernike',
                    dict(c=c_y, k=k_y, normalization_radius=float(nrr),
                         nms=nms, coefs=tuple(coefs), norm=False))

    # XY polynomial surface
    if sd.get('xyp_coefs') is not None:
        coefs = sd['xyp_coefs']
        nrr = sd.get('nrr') or 1.0
        mns = [xy_j_to_mn(j) for j in range(1, len(coefs) + 1)]
        return spec('xy',
                    dict(c=c_y, k=k_y, normalization_radius=float(nrr),
                         mns=mns, coefs=tuple(coefs)))

    # Biconic (anisotropic curvature on the two axes)
    if c_x is not None or k_x is not None:
        cx_resolved = c_x if c_x is not None else c_y
        kx_resolved = float(k_x) if k_x is not None else 0.0
        return spec('biconic',
                    dict(c_x=cx_resolved, c_y=c_y, k_x=kx_resolved, k_y=k_y))

    if sd.get('is_asphere'):
        coefs_dict = sd.get('asphere_coefs', {})
        if coefs_dict:
            n_coefs = max(coefs_dict)
            coefs = tuple(coefs_dict.get(i, 0.0)
                          for i in range(1, n_coefs + 1))
        else:
            coefs = ()
        return spec('even_asphere', dict(c=c_y, k=k_y, coefs=coefs))

    return spec('conic', dict(c=c_y, k=k_y))


def _lookup_codev_glass(glass, database):
    """Resolve a Code V GLA token, stripping a trailing _CATALOG suffix.

    Code V names glasses GLASS_CATALOG (e.g. BSM24_OHARA, N-BK7_SCHOTT).  Try
    the literal name first, then the part before the last underscore, so the
    catalog-qualified spelling still resolves against a refractiveindex.info
    page named for the bare glass.

    """
    if glass is None:
        return _materials.lookup(glass, database=database)
    try:
        return _materials.lookup(glass, database=database)
    except KeyError:
        if '_' in glass:
            return _materials.lookup(glass.rsplit('_', 1)[0], database=database)
        raise


def _glass_name(material, typ):
    """Best-effort Code V glass token for a LensData material."""
    from ..spencer_and_murty import STYPE_REFLECT
    from ..surfaces import _map_stype
    if _map_stype(typ) == STYPE_REFLECT:
        return 'REFL'
    if material is None:
        return None
    page_info = getattr(material, 'page_info', None)
    if page_info and page_info.get('page'):
        return page_info['page']
    return None  # air or an un-nameable callable -> blank (air)


def _coordbreak_seq_lines(row):
    """Code V decenter/tilt commands for a LensData CoordBreak."""
    dx, dy, dz = (float(v) for v in row.decenter)
    rz, ry, rx = (float(v) for v in row.tilt)
    lines = []
    if getattr(row, 'kind', 'basic') == 'dar':
        lines.append('DAR')
    if dx:
        lines.append(f'XDE {dx:g}')
    if dy:
        lines.append(f'YDE {dy:g}')
    if dz:
        lines.append(f'ZDE {dz:g}')
    # Code V ADE/BDE are left-handed about X/Y; invert on export.
    if rx:
        lines.append(f'ADE {-rx:g}')
    if ry:
        lines.append(f'BDE {-ry:g}')
    if rz:
        lines.append(f'CDE {rz:g}')
    return lines


def write_seq(lensdata):
    """Serialize a LensData to Code V .seq text (rotationally symmetric subset).

    Writes curvature mode (CUM), so curvatures export directly without radius
    reciprocals.  Post-reflection gaps are written with the Code V negative-
    thickness (unfolded-axis) convention -- the inverse of the import fold.
    Wavelengths export in nanometers (prysm stores microns).  Coordinate breaks
    export DEC/ADE/BDE/CDE with the Code V left-handed sign convention applied
    at this boundary only.

    """
    from ..lensdata import CoordBreak
    lines = ['LEN', 'CUM', 'DIM M']
    wvls = list(lensdata.wavelengths.values())
    if wvls:
        lines.append('WL ' + ' '.join(f'{w * 1000.0:g}' for w in wvls))
    if lensdata.epd is not None:
        lines.append(f'EPD {lensdata.epd:g}')
    if lensdata.fields:
        lines.append('YAN ' + ' '.join(f'{f.hy:g}' for f in lensdata.fields))
    lines.append('SO ; THI 1E10')

    n_refl = 0
    pending_coordbreak = None
    for row in lensdata.rows:
        if isinstance(row, CoordBreak):
            if pending_coordbreak is not None:
                raise NotImplementedError(
                    'write_seq cannot export consecutive CoordBreak rows '
                    'without an intervening surface'
                )
            pending_coordbreak = row
            continue
        from ..spencer_and_murty import STYPE_EVAL
        from ..surfaces import _map_stype
        is_eval = _map_stype(row.typ) == STYPE_EVAL
        writable_shape_or_raise(row.shape_kind, is_eval, 'write_seq')
        shape = row.build_shape()
        params = shape.params or {}
        is_refl = _glass_name(row.material, row.typ) == 'REFL'
        if is_refl:
            n_refl += 1
        sign = fold_sign(n_refl)
        thi = sign * float(row.thickness)
        if is_eval:
            lines.append('SI')
            if pending_coordbreak is not None:
                lines.extend(_coordbreak_seq_lines(pending_coordbreak))
                pending_coordbreak = None
            continue
        parts = ['S', f'CUY {params.get("c", 0.0):g}', f'THI {thi:g}']
        if params.get('k', 0.0):
            parts.insert(2, f'K {params["k"]:g}')
        glass = _glass_name(row.material, row.typ)
        if glass:
            parts.append(f'GLA {glass}')
        lines.append(' ; '.join(parts))
        if pending_coordbreak is not None:
            lines.extend(_coordbreak_seq_lines(pending_coordbreak))
            pending_coordbreak = None
    if pending_coordbreak is not None:
        raise NotImplementedError(
            'write_seq cannot export a trailing CoordBreak with no surface'
        )
    lines.append('GO')
    return '\n'.join(lines) + '\n'


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
