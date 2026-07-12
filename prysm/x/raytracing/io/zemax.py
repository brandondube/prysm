"""Zemax .zmx prescription reader."""

import math

from prysm.mathops import np

from ... import materials as _materials
from ._indexing import noll_to_nm, xy_j_to_mn
from ._common import (
    fields_from_xy,
    read_text_or_path,
    fold_sign,
    writable_shape_or_raise,
    warn_vignetting_ignored as _warn_vignetting_ignored,
    length_scale_to_mm,
    scale_length_to_mm,
    aperture_kwargs_from_radii,
    parse_float,
)
from ..lensdata import LensData
from ..system import OpticalSystem, ApertureSpec
from ._surface_spec import build_shape, surface_spec_factory


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
            val = parse_float(tokens[1])
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
    'FT': 'ft',
    'FEET': 'ft',
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
        'field_values': ([], [], 0),
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
            # 2 = paraxial image height, 3 = real image height.
            # The latter two are rejected by read_zmx; Field(kind='height')
            # represents object height, not image height.
            tokens = rest.split()
            if tokens:
                out['extras']['FTYP'] = int(tokens[0])
        else:
            out['extras'].setdefault(d, []).append(rest)
    # build Field objects from XFLN/YFLN if present
    ftype = out['extras'].get('FTYP', 0)
    out['field_values'] = (xfln, yfln, ftype)
    if xfln or yfln:
        if ftype == 0:
            out['fields'] = fields_from_xy(xfln, yfln, kind='angle',
                                           unit='deg')
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
            out['curv'] = parse_float(tokens[0]) if tokens else 0.0
        elif d == 'CONI':
            out['coni'] = parse_float(tokens[0]) if tokens else 0.0
        elif d == 'DISZ':
            out['disz'] = parse_float(tokens[0]) if tokens else 0.0
        elif d == 'GLAS':
            out['glas'] = tokens[0] if tokens else ''
        elif d == 'NMAT':
            # alternate glass spec: NMAT <name>
            out.setdefault('glas', tokens[0] if tokens else '')
        elif d == 'DIAM':
            try:
                out['diam'] = parse_float(tokens[0])
            except (IndexError, ValueError):
                pass
        elif d == 'PARM':
            # PARM <i> <value>; per-surface auxiliary parameters (e.g.,
            # asphere coefs for EVENASPH, biconic radii for BICONICX)
            if len(tokens) >= 2:
                try:
                    i = int(tokens[0])
                    v = parse_float(tokens[1])
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

def _make_spec(block, database, length_scale=1.0):
    """Build a (pose-free) SurfaceSpec from a parsed Zemax SURF block.

    Returns a _CoordinateBreak sentinel for COORDBRK pseudo-surfaces.

    """
    surf_type = block.get('type', 'STANDARD')
    c = block.get('curv', 0.0)
    k = block.get('coni', 0.0)
    glas = block.get('glas', '')
    n_callable = _materials.lookup(glas, database=database)
    spec = surface_spec_factory(n_callable, length_scale)

    if surf_type == 'STANDARD':
        return spec('conic', dict(c=c, k=k))

    if surf_type == 'EVENASPH':
        # PARM 1 = a4, PARM 2 = a6, PARM 3 = a8, ...
        coefs_dict = block.get('parm', {})
        if not coefs_dict:
            coefs = ()
        else:
            n_coefs = max(coefs_dict)
            coefs = tuple(coefs_dict.get(i, 0.0) for i in range(1, n_coefs + 1))
        return spec('even_asphere', dict(c=c, k=k, coefs=coefs))

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
        return spec('toroid',
                    dict(c_x=c_x, c_y=c_y, k_y=k_y, coefs_y=coefs_y))

    if surf_type == 'BICONICX':
        # PARM 1 = c_x; PARM 2 = k_x.  CURV = c_y, CONI = k_y
        c_x = float(block.get('parm', {}).get(1, 0.0))
        k_x = float(block.get('parm', {}).get(2, 0.0))
        return spec('biconic',
                    dict(c_x=c_x, c_y=float(c), k_x=k_x, k_y=float(k)))

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
            # no Zernike content -> degenerate to a Conic base
            return spec('conic', dict(c=c, k=k))
        max_j = max(xdat)
        nms = [noll_to_nm(j) for j in range(1, max_j + 1)]
        coefs = tuple(float(xdat.get(j, 0.0)) for j in range(1, max_j + 1))
        return spec('zernike',
                    dict(c=c, k=k, normalization_radius=float(norm_r),
                         nms=nms, coefs=coefs, norm=True))

    if surf_type == 'XYPOLY':
        p = block.get('parm', {})
        norm_r = p.get(1, 1.0)
        if norm_r == 0.0:
            norm_r = 1.0
        xdat = _parse_xdat_lines(block.get('xdat', []))
        if not xdat:
            return spec('conic', dict(c=c, k=k))
        max_j = max(xdat)
        mns = [xy_j_to_mn(j) for j in range(1, max_j + 1)]
        coefs = tuple(float(xdat.get(j, 0.0)) for j in range(1, max_j + 1))
        return spec('xy',
                    dict(c=c, k=k, normalization_radius=float(norm_r),
                         mns=mns, coefs=coefs))

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

    def tilt_decenter(self, length_scale=1.0):
        p = self.block.get('parm', {})
        decenter = (
            scale_length_to_mm(p.get(1, 0.0), length_scale),
            scale_length_to_mm(p.get(2, 0.0), length_scale),
            0.0,
        )
        # Zemax tilt order: PARM 3=Tx, 4=Ty, 5=Tz; prysm uses (rz, ry, rx)
        tilt = (p.get(5, 0.0), p.get(4, 0.0), p.get(3, 0.0))
        return tilt, decenter


# ---------- top-level reader ------------------------------------------------

def _glas_line(material):
    """'  GLAS <page>' line for a nameable non-air material, else None."""
    if material is _materials.air or material is _materials.vacuum:
        return None
    page = getattr(material, 'page_info', None)
    if page and page.get('page'):
        return f'  GLAS {page["page"]}'
    return None


def write_zmx(system):
    """Serialize an OpticalSystem to Zemax .zmx text (rotationally symmetric subset).

    Emits OBJECT (surface 0), the surface rows, and the IMAGE plane.  Post-
    reflection gaps use Zemax's negative-thickness (unfolded-axis) convention,
    the inverse of the import fold.  Coordinate breaks export as COORDBRK
    pseudo-surfaces.

    A bare LensData (no system metadata) is also accepted; aperture, stop,
    unit, and wavelengths are then simply omitted from the output.

    """
    from ..lensdata import CoordBreak
    from ..listings import surface_row_mappings
    from ..spencer_and_murty import (
        STYPE_OBJ, STYPE_REFLECT, _is_measurement_surf)
    from ..surfaces import _map_stype

    lines = ['VERS 100000 0', 'MODE SEQ']
    unit = getattr(system, 'unit', None)
    if unit:
        lines.append(f'UNIT {unit.upper()}')
    epd = getattr(system, 'epd', None)
    if epd is not None:
        lines.append(f'ENPD {epd:g}')
    stop_index = getattr(system, 'stop_index', None)
    if stop_index is not None:
        stop_surface = None
        for mapping in surface_row_mappings(system):
            if mapping['surface_index'] == stop_index:
                stop_surface = mapping['zemax_surface_number']
                break
        if stop_surface is None:
            raise ValueError(
                f'stop_index {stop_index!r} does not identify a compiled '
                'surface'
            )
        lines.append(f'STOP {stop_surface}')
    wvls = getattr(system, 'wavelengths', None)
    for w in ([] if wvls is None else wvls):
        lines.append(f'WAVL {float(w):g}')

    obj_row = next((r for r in system.rows
                    if not isinstance(r, CoordBreak)
                    and _map_stype(r.typ) == STYPE_OBJ), None)
    obj_thi = float(obj_row.thickness) if obj_row is not None else float('inf')
    disz = 'INFINITY' if not math.isfinite(obj_thi) else f'{obj_thi:g}'
    surf0 = ['SURF 0', '  TYPE STANDARD', '  CURV 0.0', f'  DISZ {disz}']
    if obj_row is not None:
        glas = _glas_line(obj_row.material)
        if glas:
            surf0.append(glas)
    lines += surf0

    surf_no = 0
    n_refl = 0
    for row in system.rows:
        if not isinstance(row, CoordBreak) \
                and _map_stype(row.typ) == STYPE_OBJ:
            continue  # OBJECT distance/medium emitted on SURF 0 above
        surf_no += 1
        if isinstance(row, CoordBreak):
            dx, dy, _ = (float(v) for v in row.decenter)
            rz, ry, rx = (float(v) for v in row.tilt)
            sign = fold_sign(n_refl)
            lines += [f'SURF {surf_no}', '  TYPE COORDBRK',
                      f'  DISZ {sign * float(row.thickness):g}',
                      f'  PARM 1 {dx:g}', f'  PARM 2 {dy:g}',
                      f'  PARM 3 {rx:g}', f'  PARM 4 {ry:g}',
                      f'  PARM 5 {rz:g}']
            continue
        is_eval = _is_measurement_surf(_map_stype(row.typ))
        writable_shape_or_raise(row.shape_kind, is_eval, 'write_zmx')
        shape = row.build_shape()
        params = shape.params or {}
        is_refl = _map_stype(row.typ) == STYPE_REFLECT
        if is_refl:
            n_refl += 1
        sign = fold_sign(n_refl)
        disz = sign * float(row.thickness)
        block = [f'SURF {surf_no}', '  TYPE STANDARD',
                 f'  CURV {params.get("c", 0.0):g}']
        if params.get('k', 0.0):
            block.append(f'  CONI {params["k"]:g}')
        block.append(f'  DISZ {disz:g}')
        if is_refl:
            block.append('  GLAS MIRROR')
        elif not is_eval:
            glas = _glas_line(row.material)
            if glas:
                block.append(glas)
        lines += block
    return '\n'.join(lines) + '\n'


def read_zmx(path_or_text, *, _is_text=False, database=None):
    """Read a Zemax .zmx text file into an OpticalSystem.

    Parameters
    ----------
    path_or_text : str
        path to a .zmx file, or (when _is_text=True) the raw text body.
    _is_text : bool
        if True, treat path_or_text as the file text rather than a path.
        Used in tests.
    database : optional
        A catalog object exposing material_for_name(name), or None to use the
        refractiveindex.info database. Required when any surface uses a real
        glass name; air, blank, and mirror surfaces do not need a database.

    Returns
    -------
    OpticalSystem

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
    unit_scale = length_scale_to_mm(header['unit'] or 'mm')

    def _gap(blk):
        d = blk.get('disz', 0.0)
        return 0.0 if not np.isfinite(d) else scale_length_to_mm(d, unit_scale)

    def _semidiameter(blk):
        return aperture_kwargs_from_radii(blk.get('diam'), unit_scale)

    fields = header['fields']
    xfln, yfln, ftype = header.get('field_values', ([], [], 0))
    if (xfln or yfln) and ftype == 1:
        raw_object_gap = parsed[0].get('disz', 0.0) if parsed else None
        if raw_object_gap is None or not np.isfinite(raw_object_gap):
            raise ValueError(
                'Zemax object-height fields require a finite object distance on '
                'SURF 0 DISZ'
            )
        fields = fields_from_xy(xfln, yfln, kind='height',
                                object_z=-raw_object_gap,
                                length_scale=unit_scale)
    elif (xfln or yfln) and ftype in (2, 3):
        raise NotImplementedError(
            'Zemax image-height fields (FTYP 2/3) are not supported by '
            'read_zmx; use angle fields or object-height fields instead'
        )
    elif (xfln or yfln) and ftype != 0:
        raise NotImplementedError(
            f'Zemax FTYP {ftype} fields are not supported by read_zmx'
        )

    ld = LensData()
    sys = OpticalSystem(
        ld,
        aperture=(ApertureSpec.epd(scale_length_to_mm(header['epd'],
                                                     unit_scale))
                  if header['epd'] is not None else None),
        fields=fields,
        wavelengths=header['wavelengths'],
        source_path=path_for_meta, source_format='zemax',
        extras=header['extras'],
    )

    # Determine which parsed block is the image surface (last real surface).
    real_indices = [i for i, blk in enumerate(parsed)
                    if not (i == 0 and blk.get('idx', i) == 0)
                    and blk.get('type', 'STANDARD') != 'COORDBRK']
    image_block_i = real_indices[-1] if real_indices else None

    # Zemax encodes post-mirror gaps as negative thicknesses on an unfolded
    # axis; LensData folds the frame at each reflection and keeps thickness
    # positive.  Convert by negating the gap once per preceding reflection.
    n_refl = 0
    for i, blk in enumerate(parsed):
        if i == 0 and blk.get('idx', i) == 0:
            # Set the auto OBJECT endpoint: object distance + object-space medium
            # (infinite conjugate keeps the default inf thickness).
            obj_spec = _make_spec(blk, database, unit_scale)
            obj_thi = _gap(blk)
            if math.isfinite(obj_thi) and obj_thi != 0.0:
                ld.object_row.thickness = obj_thi
            if obj_spec.n is not None:
                ld.object_row.material = obj_spec.n
            continue
        surf_type = blk.get('type', 'STANDARD')
        if surf_type == 'COORDBRK':
            cb = _CoordinateBreak(blk)
            tilt, decenter = cb.tilt_decenter(unit_scale)
            sign = fold_sign(n_refl)
            ld.add_coordbreak(decenter=decenter, tilt=tilt, kind='basic',
                              thickness=sign * _gap(blk))
            continue
        spec = _make_spec(blk, database, unit_scale)
        if spec.typ == 'refl':
            n_refl += 1
        sign = fold_sign(n_refl)
        thickness = sign * _gap(blk)
        aperture_kwargs = _semidiameter(blk)
        # the image surface, if flat, sets the auto IMAGE endpoint
        if i == image_block_i and spec.kind == 'conic' \
                and spec.params.get('c', 0.0) == 0.0 \
                and spec.params.get('k', 0.0) == 0.0:
            ld.image_row.thickness = thickness
            for key, val in aperture_kwargs.items():
                setattr(ld.image_row, key, val)
            continue
        ld.add(build_shape(spec), thickness=thickness,
               material=spec.n, typ=spec.typ, **aperture_kwargs)

    # translate the Zemax stop SURF number to the compiled-surface index via the
    # row<->compiled-index owner, so the OBJECT-at-0 layout isn't baked in here.
    from ..listings import surface_row_mappings
    stop_origin = header.get('stop_index_zemax')
    if stop_origin is not None:
        sys.stop_index = None
        for mapping in surface_row_mappings(sys):
            if (mapping['surface_index'] is not None
                    and mapping['zemax_surface_number'] == stop_origin):
                sys.stop_index = mapping['surface_index']
                break

    _warn_vignetting_ignored(text, 'Zemax')
    return sys
