"""Zemax .zmx (text) prescription reader.

Parses a sequential .zmx into a list of Surface objects plus header
metadata.  Supports the surface types that prysm.x.raytracing models
natively: STANDARD (conic), EVENASPH (even asphere), TOROIDAL, BICONICX,
and the COORDBRK pseudo-surface (folds into the next surface as
tilt+decenter).
ZERNSAG and XYPOLY are recognized but currently raise NotImplementedError
because mapping their PARM coefficient layouts onto prysm shape parameters is
non-trivial.

The header is parsed for: WAVL (wavelengths), ENPD (entrance pupil
diameter), STOP (stop surface index), FNUM, FTYP/XFLN/YFLN (fields),
UNIT (length units).

Returns a PrescriptionFile vanilla class carrying .surfaces (the list
ready for raytrace) plus all the metadata fields.

Out of scope (raises informative NotImplementedError):
- Multi-configuration (MNCA / MOFF) data
- Nonsequential ray-trace blocks (NSC*)
- USERSURF and other extension surface types
- Reverse / forward toggles, vignetting, polarization

For files outside this subset, supply a manually-built prescription
instead of read_zmx.

"""

from prysm.mathops import np

from .surfaces import PlaneSag, Surface
from . import materials as _materials
from ._indexing import noll_to_nm, xy_j_to_mn
from ._io_common import fields_from_xy, read_text_or_path
from ._prescription import PrescriptionFile
from ._surface_spec import SurfaceSpec, build_surface


# ---------- low-level tokenizer ---------------------------------------------

def _split_into_blocks(lines):
    """Group .zmx lines into header lines plus per-surface blocks.

    Returns (header_lines, surf_blocks) where surf_blocks is a list of
    (surf_index_int, list_of_lines) tuples in document order.  The first
    SURF line is consumed as the block header, not retained in the
    block's line list.

    """
    header = []
    blocks = []
    current = None  # (idx, [lines])
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith('SURF '):
            if current is not None:
                blocks.append(current)
            try:
                idx = int(stripped.split(None, 1)[1])
            except (IndexError, ValueError) as e:
                raise ValueError(
                    f'malformed SURF line: {line!r}'
                ) from e
            current = (idx, [])
        elif current is None:
            header.append(stripped)
        else:
            current[1].append(stripped)
    if current is not None:
        blocks.append(current)
    return header, blocks


def _directive(line):
    """Split 'TYPE STANDARD' into ('TYPE', 'STANDARD'); rest may be empty."""
    parts = line.split(None, 1)
    if len(parts) == 1:
        return parts[0].upper(), ''
    return parts[0].upper(), parts[1]


from ._io_common import parse_float as _parse_float  # noqa: E402  (kept name for callers)


def _parse_xdat_lines(lines):
    """Convert raw XDAT lines like ['1 0.5 0 0', '2 -0.1 0 0', ...] into
    a {term_idx: value} dict.

    Lines that cannot be parsed are skipped silently.

    """
    out = {}
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            idx = int(tokens[0])
            val = _parse_float(tokens[1])
            out[idx] = val
        except (ValueError, IndexError):
            pass
    return out


# ---------- header parsing --------------------------------------------------

_UNIT_MAP = {
    'MM': 'mm',
    'CM': 'cm',
    'IN': 'in',
    'INCHES': 'in',
    'M': 'm',
    'METERS': 'm',
}


def _parse_header(lines):
    """Pull the bits of the header we care about; everything else goes
    into extras."""
    out = {
        'wavelengths': [],
        'epd': None,
        'stop_index_zemax': None,  # 1-based Zemax index, translated later
        'unit': None,
        'fields': [],
        'extras': {},
    }
    xfln = []
    yfln = []
    for line in lines:
        d, rest = _directive(line)
        if d == 'WAVL':
            try:
                out['wavelengths'].append(float(rest.split()[0]))
            except (IndexError, ValueError):
                out['extras'].setdefault('WAVL_unparsed', []).append(rest)
        elif d == 'WAVM':
            # WAVM <idx> <wavelength> <weight>
            tokens = rest.split()
            if len(tokens) >= 2:
                try:
                    out['wavelengths'].append(float(tokens[1]))
                except ValueError:
                    pass
        elif d == 'ENPD':
            try:
                out['epd'] = float(rest.split()[0])
            except (IndexError, ValueError):
                pass
        elif d == 'STOP':
            try:
                out['stop_index_zemax'] = int(rest.split()[0])
            except (IndexError, ValueError):
                pass
        elif d == 'UNIT':
            t = rest.split()
            if t:
                out['unit'] = _UNIT_MAP.get(t[0].upper(), t[0].lower())
        elif d == 'XFLN':
            xfln = [float(x) for x in rest.split() if x]
        elif d == 'YFLN':
            yfln = [float(y) for y in rest.split() if y]
        elif d == 'FTYP':
            # FTYP <type> ... type 0 = angle, 1 = object height,
            # 2 = paraxial image height, 3 = real image height
            tokens = rest.split()
            if tokens:
                out['extras']['FTYP'] = int(tokens[0])
        else:
            out['extras'].setdefault(d, []).append(rest)
    # build Field objects from XFLN/YFLN if present
    ftype = out['extras'].get('FTYP', 0)
    if xfln or yfln:
        kind = 'angle' if ftype == 0 else 'height'
        out['fields'] = fields_from_xy(xfln, yfln, kind=kind, unit='deg')
    return out


# ---------- per-surface parsing ---------------------------------------------

def _parse_block(idx, body_lines):
    """Reduce a SURF block to a dict of normalized fields.

    Returns a dict with keys present only when the directive appeared;
    callers use .get with sensible defaults.

    """
    out = {'idx': idx, 'parm': {}}
    for line in body_lines:
        d, rest = _directive(line)
        tokens = rest.split()
        if d == 'TYPE':
            if tokens:
                out['type'] = tokens[0].upper()
        elif d == 'CURV':
            out['curv'] = _parse_float(tokens[0]) if tokens else 0.0
        elif d == 'CONI':
            out['coni'] = _parse_float(tokens[0]) if tokens else 0.0
        elif d == 'DISZ':
            out['disz'] = _parse_float(tokens[0]) if tokens else 0.0
        elif d == 'GLAS':
            out['glas'] = tokens[0] if tokens else ''
        elif d == 'NMAT':
            # alternate glass spec: NMAT <name>
            out.setdefault('glas', tokens[0] if tokens else '')
        elif d == 'DIAM':
            try:
                out['diam'] = _parse_float(tokens[0])
            except (IndexError, ValueError):
                pass
        elif d == 'PARM':
            # PARM <i> <value>; per-surface auxiliary parameters (e.g.,
            # asphere coefs for EVENASPH, biconic radii for BICONICX)
            if len(tokens) >= 2:
                try:
                    i = int(tokens[0])
                    v = _parse_float(tokens[1])
                    out['parm'][i] = v
                except ValueError:
                    pass
        elif d == 'XDAT':
            # extra surface data (Zernike sag coefficients, etc.); record
            # raw for surface-type-specific consumers
            out.setdefault('xdat', []).append(rest)
        elif d in ('STOP',):
            out['is_stop'] = True
        elif d == 'COMM':
            out['comment'] = rest
        elif d in ('MEMA', 'CTGT', 'CONF', 'HIDE', 'MIRR', 'COAT'):
            # known but ignored directives
            pass
        else:
            # silently ignore unknown directives; record them for diag
            out.setdefault('unknown', []).append(line)
    return out


# ---------- block -> Surface ------------------------------------------------

def _make_surface(block, vertex_z, database):
    """Build one prysm Surface from a parsed Zemax SURF block."""
    surf_type = block.get('type', 'STANDARD')
    c = block.get('curv', 0.0)
    k = block.get('coni', 0.0)
    glas = block.get('glas', '')
    n_callable = _materials.lookup(glas, database=database)
    is_mirror = (n_callable is _materials.MIRROR)
    typ = 'refl' if is_mirror else 'refr'
    if not is_mirror and (glas == '' or glas is None):
        # explicit air on the post-medium: typ remains 'refr' with n=air
        # (matches Zemax convention: missing GLAS == air after surface)
        pass
    n_arg = None if is_mirror else n_callable
    P = [0.0, 0.0, float(vertex_z)]

    if surf_type == 'STANDARD':
        return build_surface(SurfaceSpec('conic', typ, P, n_arg,
                                         dict(c=c, k=k)))

    if surf_type == 'EVENASPH':
        # PARM 1 = a4, PARM 2 = a6, PARM 3 = a8, ...
        coefs_dict = block.get('parm', {})
        if not coefs_dict:
            coefs = ()
        else:
            n_coefs = max(coefs_dict)
            coefs = tuple(coefs_dict.get(i, 0.0) for i in range(1, n_coefs + 1))
        return build_surface(SurfaceSpec('even_asphere', typ, P, n_arg,
                                         dict(c=c, k=k, coefs=coefs)))

    if surf_type == 'TOROIDAL':
        # PARM 1 = radius_of_rotation (= 1/c_x); CURV = c_y, CONI = k_y
        # PARM 2.. = aspheric coefs in y
        rot = block.get('parm', {}).get(1, None)
        if rot is None or rot == 0.0:
            raise ValueError(
                f'TOROIDAL surface {block["idx"]} missing PARM 1 '
                '(radius of rotation)'
            )
        c_x = 1.0 / float(rot)
        c_y = float(c)
        k_y = float(k)
        coefs_dict = block.get('parm', {})
        # PARM 2..N = a4_y, a6_y, ...
        if len(coefs_dict) > 1:
            n_coefs = max(coefs_dict) - 1
            coefs_y = tuple(coefs_dict.get(i + 1, 0.0)
                            for i in range(1, n_coefs + 1))
        else:
            coefs_y = ()
        return build_surface(SurfaceSpec(
            'toroid', typ, P, n_arg,
            dict(c_x=c_x, c_y=c_y, k_y=k_y, coefs_y=coefs_y)))

    if surf_type == 'BICONICX':
        # PARM 1 = c_x; PARM 2 = k_x.  CURV = c_y, CONI = k_y
        c_x = float(block.get('parm', {}).get(1, 0.0))
        k_x = float(block.get('parm', {}).get(2, 0.0))
        return build_surface(SurfaceSpec(
            'biconic', typ, P, n_arg,
            dict(c_x=c_x, c_y=float(c), k_x=k_x, k_y=float(k))))

    if surf_type == 'ZERNSAG':
        p = block.get('parm', {})
        norm_r = p.get(1)
        if norm_r is None or norm_r == 0.0:
            raise ValueError(
                f'ZERNSAG surface {block["idx"]} missing PARM 1 '
                '(normalization radius)'
            )
        xdat = _parse_xdat_lines(block.get('xdat', []))
        if not xdat:
            # fallback: some legacy files store Zernike coefs in PARM
            # 4..N (PARM 1..3 reserved for norm radius / decenter /
            # extrapolate flags).  Pull any PARM with key >= 4.
            xdat = {(k - 3): v for k, v in p.items()
                    if k >= 4 and v != 0.0}
        if not xdat:
            # no Zernike content -> degenerate to a Conic base
            return build_surface(SurfaceSpec('conic', typ, P, n_arg,
                                             dict(c=c, k=k)))
        max_j = max(xdat)
        nms = [noll_to_nm(j) for j in range(1, max_j + 1)]
        coefs = tuple(float(xdat.get(j, 0.0)) for j in range(1, max_j + 1))
        return build_surface(SurfaceSpec(
            'zernike', typ, P, n_arg,
            dict(c=c, k=k, normalization_radius=float(norm_r),
                 nms=nms, coefs=coefs, norm=True)))

    if surf_type == 'XYPOLY':
        p = block.get('parm', {})
        norm_r = p.get(1, 1.0)
        if norm_r == 0.0:
            norm_r = 1.0
        xdat = _parse_xdat_lines(block.get('xdat', []))
        if not xdat:
            # legacy fallback: coefficients in PARM 2..N (skip PARM 1 =
            # norm radius)
            xdat = {(k - 1): v for k, v in p.items() if k >= 2 and v != 0.0}
        if not xdat:
            return build_surface(SurfaceSpec('conic', typ, P, n_arg,
                                             dict(c=c, k=k)))
        max_j = max(xdat)
        mns = [xy_j_to_mn(j) for j in range(1, max_j + 1)]
        coefs = tuple(float(xdat.get(j, 0.0)) for j in range(1, max_j + 1))
        return build_surface(SurfaceSpec(
            'xy', typ, P, n_arg,
            dict(c=c, k=k, normalization_radius=float(norm_r),
                 mns=mns, coefs=coefs)))

    if surf_type == 'COORDBRK':
        # not a real surface; return a sentinel handled by the caller
        return _CoordinateBreak(block)

    raise NotImplementedError(
        f'Zemax surface type {surf_type!r} not supported by read_zmx.  '
        'Supported: STANDARD, EVENASPH, TOROIDAL, BICONICX, ZERNSAG, '
        'XYPOLY, COORDBRK (folded into the next surface).'
    )


class _CoordinateBreak:
    """Internal sentinel: SURF k of TYPE COORDBRK is not a physical surface;
    it perturbs the next surface by a tilt+decenter pair encoded in PARM
    1..6 (decenter X, decenter Y, tilt about X, tilt about Y, tilt about
    Z, order toggle).

    """

    __slots__ = ('block',)

    def __init__(self, block):
        self.block = block

    def tilt_decenter(self):
        p = self.block.get('parm', {})
        decenter = (p.get(1, 0.0), p.get(2, 0.0), 0.0)
        # Zemax tilt order: PARM 3=Tx, 4=Ty, 5=Tz; prysm uses (rz, ry, rx)
        tilt = (p.get(5, 0.0), p.get(4, 0.0), p.get(3, 0.0))
        return tilt, decenter


# ---------- top-level reader ------------------------------------------------

def read_zmx(path_or_text, *, _is_text=False, database=None):
    """Read a Zemax .zmx text file into a PrescriptionFile.

    Parameters
    ----------
    path_or_text : str
        path to a .zmx file, or (when _is_text=True) the raw text body.
    _is_text : bool
        if True, treat path_or_text as the file text rather than a path.
        Used in tests.
    database : refractivesqlite.Database, optional
        Required when any surface uses a real glass name. Air, blank, and
        mirror surfaces do not need a database.

    Returns
    -------
    PrescriptionFile

    """
    text, path_for_meta = read_text_or_path(path_or_text, is_text=_is_text)
    lines = text.splitlines()
    header_lines, surf_blocks = _split_into_blocks(lines)
    header = _parse_header(header_lines)

    # Walk the surfaces in order.  In Zemax, surface 0 is OBJECT (light
    # source); the last surface is IMAGE.  The vertex-z of surface k is
    # the cumulative sum of DISZ from surface 0 through k-1.
    if not surf_blocks:
        raise ValueError('no surfaces found in .zmx text')

    # parse each block
    parsed = [_parse_block(idx, body) for idx, body in surf_blocks]

    # build vertex-z trail, skipping COORDBRK-induced perturbations from
    # the running z (Zemax COORDBRK does contribute its DISZ to z; we
    # honour that)
    z = 0.0
    vertex_z = []
    for blk in parsed:
        vertex_z.append(z)
        d = blk.get('disz', 0.0)
        if not np.isfinite(d):
            # OBJECT at infinity: DISZ is INFINITY on surface 0 typically
            d = 0.0
        z += d

    # First pass: build raw surfaces, applying any COORDBRK that
    # immediately precedes a real surface.  Skip OBJECT (idx 0).
    surfaces = []
    pending_tilt = None
    pending_decenter = None
    surface_origin_idx = []  # parallel to surfaces; gives the original Zemax idx
    for i, blk in enumerate(parsed):
        if blk.get('idx', i) == 0 and i == 0:
            # OBJECT surface: skip but record DISZ already in vertex_z[1]
            continue
        surf_type = blk.get('type', 'STANDARD')
        if surf_type == 'COORDBRK':
            cb = _CoordinateBreak(blk)
            t, d = cb.tilt_decenter()
            pending_tilt = t
            pending_decenter = d
            continue
        s = _make_surface(blk, vertex_z[i], database)
        if pending_tilt is not None or pending_decenter is not None:
            # apply by mutating P / R via Surface's own pathway: rebuild
            # with tilt= and decenter= would require re-issuing the
            # factory call; simplest is to mutate after the fact using
            # the same composition rule in surfaces._apply_tilt_decenter.
            from .surfaces import _apply_tilt_decenter, _none_or_rotmat
            R = _none_or_rotmat(s.R)
            new_P, new_R = _apply_tilt_decenter(
                s.P, R, pending_tilt, pending_decenter, tilt_radians=False,
            )
            s.P = new_P
            s.R = new_R
            pending_tilt = None
            pending_decenter = None
        # if the LAST real surface has no GLAS (just air post-medium) and
        # is the IMAGE plane, demote to typ='eval'
        surfaces.append(s)
        surface_origin_idx.append(blk.get('idx', i))

    # the last surface is IMAGE — rebuild as a proper Plane(typ='eval')
    # if it is flat (c=0, k=0), so it gets the closed-form plane intersect.
    if surfaces:
        last = surfaces[-1]
        is_flat = (last.params is None
                   or (last.params.get('c', 0.0) == 0.0
                       and last.params.get('k', 0.0) == 0.0))
        if is_flat:
            last_z = float(last.P[2])
            surfaces[-1] = Surface(shape=PlaneSag(), typ='eval',
                                   P=[0.0, 0.0, last_z])

    # translate Zemax stop index (1-based, refers to Zemax SURF idx) to
    # the index inside our `surfaces` list (which has OBJECT stripped)
    stop_origin = header.get('stop_index_zemax')
    stop_index = None
    if stop_origin is not None:
        try:
            stop_index = surface_origin_idx.index(stop_origin)
        except ValueError:
            stop_index = None

    return PrescriptionFile(
        surfaces=surfaces,
        wavelengths=header['wavelengths'],
        epd=header['epd'],
        stop_index=stop_index,
        fields=header['fields'],
        unit=header['unit'],
        source_path=path_for_meta,
        source_format='zemax',
        extras=header['extras'],
    )
