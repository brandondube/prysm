"""File readers for various commercial instruments."""
from io import StringIO, IOBase
import re
import struct
import codecs
import datetime
import calendar
import shutil
import warnings

from pathlib import Path

import numpy as truenp

from .conf import config
from .mathops import np


def read_file_stream_or_path(path_or_file):
    try:
        with codecs.open(path_or_file, mode='r', encoding='cp1252') as fid:
            data = codecs.encode(fid.read(), 'utf-8').decode('utf-8')
    except (FileNotFoundError, TypeError):  # FNF -- file object, TypeError -- file_like
        try:
            path_or_file.seek(0)
            raw = path_or_file.read()
            data = codecs.encode(raw, 'utf-8').decode('utf-8')
        except TypeError:  # opened in bytes mode
            data = raw.decode('cp1252')
        except AttributeError:
            data = path_or_file  # TODO: avoid duplicate
    except (AttributeError, UnicodeDecodeError):
        data = path_or_file

    return data


def is_mtfvfvf_file(file):
    """Read MTF vs Field vs Focus data from a Trioptics .txt dump.

    Parameters
    ----------
    file : str or path_like or file_like
        file to read from, if string of file body, must provide filename

    Returns
    -------
    boolean : bool
        if the file is an MTFvFvF file
    data : str
        contents of the file

    """
    data = read_file_stream_or_path(file)
    if data.startswith('ImgHeight'):
        return True, data
    else:
        return False, data


def read_trioptics_mtfvfvf(file, filename=None):
    """Read MTF vs Field vs Focus data from a Trioptics .txt dump.

    Parameters
    ----------
    file : str or path_like or file_like
        file to read from, if string of file body, must provide filename
    filename : str, optional
        name of file; used to select tan/sag if file is given as contents

    Returns
    -------
    MTFvFvF
        MTF vs Field vs Focus object

    """
    if filename is None:
        with open(file, 'r') as fid:
            lines = fid.readlines()
    else:
        lines = file.splitlines()
        file = filename

    if str(file)[-7:-4] == 'Tan':
        azimuth = 'Tan'
    else:
        azimuth = 'Sag'

    imghts, objangs, focusposes, mtfs = [], [], [], []
    for meta, data in zip(lines[0::2], lines[1::2]):  # iterate 2 lines at a time
        metavalues = meta.split()
        imght, objang, focuspos, freqpitch = metavalues[1::2]
        mtf_raw = data.split()[1:]  # first element is "MTF"
        mtf = np.asarray(mtf_raw, dtype=config.precision)
        imghts.append(imght)
        objangs.append(objang)
        focusposes.append(focuspos)
        mtfs.append(mtf)

    focuses = np.unique(np.asarray(focusposes, dtype=config.precision))
    focuses = (focuses - np.mean(focuses)) * 1e3
    imghts = np.unique(np.asarray(imghts, dtype=config.precision))
    freqs = np.arange(len(mtfs[0]), dtype=config.precision) * float(freqpitch)
    data = np.swapaxes(np.asarray(mtfs).reshape(len(focuses), len(imghts), len(freqs)), 0, 1)
    return {
        'data': data,
        'focus': focuses,
        'field': imghts,
        'freq': freqs,
        'azimuth': azimuth
    }


def read_trioptics_mtf_vs_field(file, metadata=False):
    """Read tangential and sagittal MTF data from a Trioptics .mht file.

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : bool
        whether to also extract and return metadata

    Returns
    -------
    dict
        dictionary with keys of freq, field, tan, sag

    """
    warnings.warn('this function will dispatch to either read_trioptics_mtf_vs_field_mtflab_v4, or _v5 in v0.20.  In v0.19, it always uses _v4.')
    return read_trioptics_mtf_vs_field_mtflab_v4(file, metadata=metadata)


def read_trioptics_mtf_vs_field_mtflab_v4(file, metadata=False):
    """Read tangential and sagittal MTF data from a Trioptics .mht file.  Compatible with MTF-Lab v4.

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : bool
        whether to also extract and return metadata

    Returns
    -------
    dict
        dictionary with keys of freq, field, tan, sag

    """
    warnings.warn('this function will dispatch to either read_trioptics_mtf_vs_field_mtflab_v4, or _v5 in v0.20.  In v0.19, it always uses _v4.')
    data = read_file_stream_or_path(file)
    data = data[:len(data)//10]  # only search in a subset of the file for speed

    # compile a pattern that will search for the image heights in the file and extract
    fields_pattern = re.compile('MTF=09(.*?)Legend=09', flags=re.DOTALL)
    fields = fields_pattern.findall(data)[0]  # two copies, only need 1st

    # make a pattern that will search for and extract the tan and sag MTF data.  The match will
    # return two copies; one for vs imght, one for vs angle.  Only keep half the matches.
    tan_pattern = re.compile(r'Tan(.*?)=97', flags=re.DOTALL)
    sag_pattern = re.compile(r'Sag(.*?)=97', flags=re.DOTALL)
    tan, sag = tan_pattern.findall(data), sag_pattern.findall(data)
    endpt = len(tan) // 2
    tan, sag = tan[:endpt], sag[:endpt]

    # now extract the freqs from the tan data
    freqs = np.asarray([float(s.split('(')[0][1:]) for s in tan])

    # lastly, extract the floating point tan and sag data
    # also take fields, to the 4th decimal place (nearest .1um)
    # reformat T/S to 2D arrays with indices of (freq, field)
    tan = np.asarray([s.split('=09')[1:-1] for s in tan], dtype=config.precision)
    sag = np.asarray([s.split('=09')[1:-1] for s in sag], dtype=config.precision)
    fields = np.asarray(fields.split('=09')[0:-1], dtype=config.precision).round(4)
    res = {
        'freq': freqs,
        'field': fields,
        'tan': tan,
        'sag': sag,
    }
    if metadata is True:
        return {**res, **parse_trioptics_metadata(data)}
    else:
        return res


def read_trioptics_mtf_vs_field_mtflab_v5(file_contents, metadata=False):
    """Read tangential and sagittal MTF data from a Trioptics .mht file.  Compatible with MTF-Lab v5.

    Parameters
    ----------
    file_contents : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : bool
        whether to also extract and return metadata

    Returns
    -------
    dict
        dictionary with keys of freq, field, tan, sag

    """
    if metadata:
        mdata = parse_trioptics_metadata_mtflab_v5(file_contents)

    end = file_contents.find('<!-- close certificate table -->')
    file_contents = file_contents[:end]

    # now chunk out the first table and get our image heights
    start = file_contents.find('<!--  begin table caption -->')
    end = file_contents.find('<!-- end table caption -->')
    image_heights = []
    body = file_contents[start+29:end]  # 29 = len of begin text
    body = body.splitlines()[8:-2]  # first, last few rows are noise
    for row in body:
        value = row.split('>', 1)[1].split('<')[0]
        image_heights.append(float(value))

    # now chunk out the second, which we parse a little differently
    file_contents = file_contents[end:]
    start = file_contents.find('<!-- begin measurement data -->')
    end = file_contents.find('<!-- end measurement data -->')
    file_contents = file_contents[start+31:end]  # 31 is len of begin
    # now file_contents is the text of the table and a little noise.
    # set up parsed tables...
    tan = []
    sag = []
    freqs = []
    rows = file_contents.split('<tr ')[1:]
    for row in rows:
        cells = row.split('<td')[1:-1]  # first, last are garbage
        # first cell is azimuth and frequency, rest are MTF vs Field
        az, freq = cells[0].split('>', 1)[1].split('<')[0].split()
        freq = float(freq.split('(')[0])
        if az == 'Sag':
            target = sag
        else:
            target = tan

        tmp = []
        for cell in cells[1:]:  # first, last cells are trash
            value = cell.split('>', 1)[1].split('<')[0]
            tmp.append(float(value))

        target.append(tmp)
        if freq not in freqs:
            freqs.append(freq)

    data = {
        'tan':  np.asarray(tan, dtype=config.precision),
        'sag': np.asarray(sag, dtype=config.precision),
        'field': np.asarray(image_heights, dtype=config.precision),
        'freq': np.asarray(freqs, dtype=config.precision),
    }
    if metadata:
        return {**data, **mdata}
    else:
        return data


def read_trioptics_mtf(file, metadata=False):
    """Read MTF data from a Trioptics data file.

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : bool
        whether to also extract and return metadata

    Returns
    -------
    dict
        dictionary with keys focus, freq, tan, sag
        if metadata=True, also has keys in the return of
        io.parse_trioptics_metadata.

    """
    data = read_file_stream_or_path(file)
    data = data[:len(data)//10]

    # compile regex scanners to grab wavelength, focus, and frequency information
    # in addition to the T, S MTF data.
    # lastly, compile a scanner to cut the file after the end of the "MTF Sagittal" scanner
    focus_scanner = re.compile(r'Focus Position  : (\-?\d+\.\d+) mm')
    data_scanner = re.compile(r'\r\n(\d+\.?\d?)=09\r\n(\d+\.\d+)=09')
    sag_scanner = re.compile(r'Measurement Table: MTF vs. Frequency \( Sagittal \)')
    blockend_scanner = re.compile(r'  _____ =20')

    sagpos, cutoff = sag_scanner.search(data).end(), None
    for blockend in blockend_scanner.finditer(data):
        if blockend.end() > sagpos and cutoff is None:
            cutoff = blockend.end()

    # get focus and wavelength
    focus_pos = float(focus_scanner.search(data).group(1))

    # simultaneously grab frequency and MTF
    result = data_scanner.findall(data[:cutoff])
    freqs, mtfs = [], []
    for dat in result:
        freqs.append(float(dat[0]))
        mtfs.append(dat[1])

    breakpt = len(mtfs) // 2
    t = np.asarray(mtfs[:breakpt], dtype=config.precision)
    s = np.asarray(mtfs[breakpt:], dtype=config.precision)
    freqs = tuple(freqs[:breakpt])

    res = {
        'focus': focus_pos,
        'freq': freqs,
        'tan': t,
        'sag': s,
    }
    if metadata is True:
        return {**res, **parse_trioptics_metadata(data)}
    else:
        return res


def parse_trioptics_metadata(file_contents):
    """Read metadata from the contents of a Trioptics .mht file.

    Parameters
    ----------
    file_contents : str
        contents of a .mht file.

    Returns
    -------
    dict
        dictionary with keys:
            - operator
            - time
            - sample_id
            - instrument
            - instrument_sn
            - collimator
            - wavelength
            - efl
            - obj_angle
            - focus_pos
            - azimuth

    """
    warnings.warn('this function will dispatch to either parse_trioptics_metadata_mtflab_v4, or _v5 in v0.20.  In v0.19, it always uses _v4.')
    return parse_trioptics_metadata_mtflab_v4(file_contents)


def parse_trioptics_metadata_mtflab_v4(file_contents):
    """Read metadata from the contents of a Trioptics .mht file.  Compatible with MTF-Lab v4.

    Parameters
    ----------
    file_contents : str
        contents of a .mht file.

    Returns
    -------
    dict
        dictionary with keys:
            - operator
            - time
            - sample_id
            - instrument
            - instrument_sn
            - collimator
            - wavelength
            - efl
            - obj_angle
            - focus_pos
            - azimuth

    """
    data = file_contents[750:1500]  # skip large section to make regex faster

    operator_scanner = re.compile(r'Operator         : (\S*)')
    time_scanner = re.compile(r'Time/Date        : (\d{2}:\d{2}:\d{2}\s*\w*\s*\d*,\s*\d*)')
    sampleid_scanner = re.compile(r'Sample ID        : (.*)')
    instrument_sn_scanner = re.compile(r'Instrument S/N   : (\S*)')

    collimatorefl_scanner = re.compile(r'EFL \(Collimator\): (\d*) mm')
    wavelength_scanner = re.compile(r'Wavelength      : (\d+) nm')
    sampleefl_scanner = re.compile(r'EFL \(Sample\)    : (\d*\.\d*) mm')
    objangle_scanner = re.compile(r'Object Angle    : (-?\d*\.\d*) =B0')
    focuspos_scanner = re.compile(r'Focus Position  : (\d*\.\d*) mm')
    azimuth_scanner = re.compile(r'Sample Azimuth  : (-?\d*\.\d*) =B0')

    operator = operator_scanner.search(data).group(1)
    time = time_scanner.search(data).group(1)
    hms, month, day, year = time.split()
    year, day = int(year), int(day[:-1])
    month_num = list(calendar.month_name).index(month)
    h, m, s = hms.split(':')
    h, m, s = (int(str_) for str_ in [h, m, s])
    timestamp = datetime.datetime(year=year, month=month_num, day=day, hour=h, minute=m, second=s)
    sampleid = sampleid_scanner.search(data).group(1).strip()
    instrument_sn = instrument_sn_scanner.search(data).group(1)

    collimator_efl = float(collimatorefl_scanner.search(data).group(1))
    wavelength = float(wavelength_scanner.search(data).group(1)) / 1e3  # nm to um
    sample_efl = float(sampleefl_scanner.search(data).group(1))
    obj_angle = float(objangle_scanner.search(data).group(1))
    focus_pos = float(focuspos_scanner.search(data).group(1))
    azimuth = float(azimuth_scanner.search(data).group(1))
    return {
        'operator': operator,
        'time': timestamp,
        'sample_id': sampleid,
        'instrument': 'Trioptics ImageMaster HR',
        'instrument_sn': instrument_sn,
        'collimator': collimator_efl,
        'wavelength': wavelength,
        'efl': sample_efl,
        'fno': None,
        'obj_angle': obj_angle,
        'focus_pos': focus_pos,
        'azimuth': azimuth,
    }


def parse_trioptics_metadata_mtflab_v5(file_contents):
    """Read metadata from the contents of a Trioptics .mht file.  Compatible with MTF-Lab v5.

    Parameters
    ----------
    file_contents : str
        contents of a .mht file.

    Returns
    -------
    dict
        dictionary with keys:
            - operator
            - time
            - sample_id
            - instrument
            - instrument_sn
            - collimator
            - wavelength
            - efl
            - obj_angle
            - focus_pos
            - azimuth

    """
    # get the first header block, there are two...
    top = file_contents.find('<pre>')
    bottom = file_contents.find('</pre>', top)
    body = file_contents[top+5:bottom].splitlines()  # 5 is len of <pre>
    sep = ': '

    company = body[0].split(sep)[-1].strip()
    operator = body[1].split(sep)[-1].strip()
    timestamp = body[2].split(sep)[-1].strip()
    timestamp = datetime.datetime.strptime(timestamp, '%H:%M:%S  %B %d, %Y')
    sampleid = body[3].split(sep)[-1].strip()
    instrument_sn = body[8].split(sep)[-1].strip()

    # now the second block
    top = file_contents.find('<pre>', bottom)
    bottom = file_contents.find('</pre>', top)
    body = file_contents[top+5:bottom].splitlines()  # 5 is len of <pre>

    # EFL (Collimator)     : 300 mm => 300 mm => [300, mm] => float(300)
    collimator_efl = float(body[1].split(sep)[-1].strip().split(' ')[0])
    wavelength = body[2].split(sep)[-1].strip()

    # EFL (Sample)        : 26.4664 mm => 20.4664 mm => [20.4664, mm] => float(20.4664)
    efl = float(body[3].split(sep)[-1].split()[0].strip())
    fno = float(body[4].split(sep)[-1].split('=')[0])
    obj_angle = float(body[5].split(sep)[-1].split()[0])
    focus_pos = float(body[6].split(sep)[-1].split()[0])
    azimuth = float(body[7].split(sep)[-1].split()[0])
    efl, fno, obj_angle, focus_pos, azimuth
    meta = {
        'company': company,
        'operator': operator,
        'timestamp': timestamp,
        'sample_id': sampleid,
        'instrument': 'Trioptics ImageMaster',
        'instrument_sn': instrument_sn,
        'collimator': collimator_efl,
        'wavelength': wavelength,
        'efl': efl,
        'fno': fno,
        'obj_angle': obj_angle,
        'focus_pos': focus_pos,
        'azimuth': azimuth,
    }
    return meta


def identify_trioptics_measurement_type(file):
    """Identify type of measurement in a Trioptics .mht file.

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object

    Returns
    -------
    program : str
        measurement type
    data : str
        contents of the file

    """
    data = read_file_stream_or_path(file)
    data_parse = data[750:1500]
    measurement_type_scanner = re.compile(r'Measure Program  : (.*)')
    program = measurement_type_scanner.search(data_parse).group(1).strip()
    return program, data


TRIOPTICS_SWITCHBOARD = {
    'MTF vs. Field': read_trioptics_mtf_vs_field,
    'Distortion': NotImplemented,
    'Axial Color': NotImplemented,
    'Lateral Color': NotImplemented,
}


def read_any_trioptics_mht(file, metadata=False):
    """Read any Trioptics .mht certificate (MTF vs Field, Distortion, etc).

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : bool
        whether to also extract and return metadata

    Returns
    -------
    dict
        dictionary with appropriate keys.  If metadata=True, also has keys in
        the return of io.parse_trioptics_metadata.

    """
    type_, data = identify_trioptics_measurement_type(file)
    return type_, TRIOPTICS_SWITCHBOARD[type_](data, metadata=metadata)


def read_mtfmapper_sfr_single(file, pixel_pitch=None):
    """Read an MTF Mapper SFR (MTF) file generated by the -f flag with --single-roi.

    Notes
    -----
    This reads a "raw_sfr_values.txt" file, not an "edge_sfr_values.txt" file.

    Parameters
    ----------
    file : str or path_like or file_like
        contents of a file, path_like to the file, or file object
    pixel_pitch : float
        center-to-center pixel spacing, in microns

    Returns
    -------
    numpy.ndarray
        spatial_frequencies
    numpy.ndarray
        mtf

    """
    data = read_file_stream_or_path(file)
    floats = [float(d) for d in data.splitlines()[0].split(' ')[:-1]]
    edge_angle, *mtf = floats
    mtf = np.asarray(mtf)
    freqs = np.arange(len(mtf)) / 64
    if pixel_pitch is not None:  # convert cy/px to cy/mm
        freqs /= (pixel_pitch / 1e3)

    return freqs, mtf


def read_zygo_datx(file):
    """Read a zygo datx file.

    Parameters
    ----------
    file : path_like
        location of a file

    Returns
    -------
    dict
        dictionary with keys phase, intensity, meta

    Raises
    ------
    ImportError
        h5py unavailable, required dependency for this

    """
    import h5py

    # create a handle to the h5 file
    with h5py.File(file, 'r') as f:
        # cast intensity down to int16, saves memory and Zygo doesn't use cameras >> 16-bit
        try:
            intens_block = list(f['Data']['Intensity'].keys())[0]
            intensity = f['Data']['Intensity'][intens_block][()].astype(np.uint16)
        except (KeyError, OSError):
            intensity = None

        # load phase
        # find the phase array's H5 group
        phase_key = list(f['Data']['Surface'].keys())[0]
        phase_obj = f['Data']['Surface'][phase_key]

        # get a little metadata
        no_data = phase_obj.attrs['No Data'][0]
        wvl = phase_obj.attrs['Wavelength'][0] * 1e9  # Zygo stores wavelength in meters, we want output in nanometers
        punit = phase_obj.attrs['Unit'][0]
        if isinstance(punit, bytes):
            punit = punit.decode('UTF-8')
        scale_factor = phase_obj.attrs['Interferometric Scale Factor']
        obliquity = phase_obj.attrs['Obliquity Factor']
        # get the phase and process it as required
        phase = phase_obj[()]
        # step 1, flip (above)
        # step 2, clip the nans
        # step 3, convert punit to nm
        phase[phase >= no_data] = np.nan
        if punit == 'Fringes':
            # the usual conversion per malacara
            phase = phase * obliquity * scale_factor * wvl
        elif punit == 'NanoMeters':
            pass
        else:
            raise ValueError("datx file does not use expected phase unit, contact the prysm author with a sample file to resolve")

        # now get attrs
        attrs = f['Attributes']
        key = list(attrs)[-1]
        attrs = attrs[key].attrs
        meta = {}
        for key, value in attrs.items():
            if key.endswith('Unit'):
                continue  # do not need unit keys, units implicitly understood.

            if key.startswith("Data Context."):
                key = key[len("Data Context."):]

            if key.startswith("Data Attributes."):
                key = key[len("Data Attributes."):]
            if key.endswith('Value'):
                key = key[:-5]  # strip value from key
            if key.endswith(':'):
                key = key[:-1]
            if key == 'Resolution':
                key = 'Lateral Resolution'
            elif key in ['Property Bag List', 'Group Number', 'TextCount']:
                continue  # h5py particulars
            if value.dtype == 'object':
                value = value[0]
                if isinstance(value, bytes):
                    value = value.decode('UTF-8')
            elif value.dtype in ['uint8', 'int32']:
                value = int(value[0])
            elif value.dtype in ['float64']:
                value = float(value[0])
            else:
                continue  # compound items, h5py objects that do not map nicely to primitives

            meta[key] = value

    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }


ZYGO_INVALID_PHASE = 2147483640
ZYGO_ENC = 'utf-8'  # may be ASCII, cp1252...
ZYGO_PHASE_RES_FACTORS = {
    0: 4096,    # 12-bit
    1: 32768,   # 15-bit
    2: 131072,  # 17-bit
}


def read_zygo_dat(file, multi_intensity_action='first'):
    """Read the contents of a zygo binary (.dat) file.

    Parameters
    ----------
    file : path_like
        path to a file
    multi_intensity_action : str, {'avg', 'first', 'last'}
        action to take when handling multiple intensitiy frames, only avg is valid at this time

    Returns
    -------
    dict
        dictionary with keys: phase, intensity, meta

    """
    with open(file, 'rb') as fid:
        contents = fid.read()

    meta = read_zygo_metadata(contents)
    iw, ih, ib = meta['ac']['width'], meta['ac']['height'], meta['ac']['n_buckets']
    if ib == 0:
        ib = 1
    ilen = iw * ih * ib  # intensity
    pw, ph = meta['cn']['width'], meta['cn']['height']
    plen = pw * ph  # phase
    header_len = meta['header']['size']

    intensity = np.frombuffer(contents, offset=header_len, count=ilen, dtype=np.uint16).reshape((ib, ih, iw))
    if multi_intensity_action.lower() == 'avg':
        intensity = intensity.mean(axis=0)
    elif multi_intensity_action.lower() == 'first':
        intensity = intensity[0]
    elif multi_intensity_action.lower() == 'last':
        intensity = intensity[-1]
    else:
        raise ValueError(f'multi_intensity_action {multi_intensity_action} not among valid options of avg, first, last.')

    # little-endian camera data, not sure if always need to byteswap, may break for some users...
    phase_raw = np.frombuffer(contents, offset=header_len + ilen * 2, count=plen, dtype=np.int32)
    phase = phase_raw.copy().byteswap(True).astype(config.precision).reshape((ph, pw))
    phase[phase >= ZYGO_INVALID_PHASE] = np.nan
    phase *= (meta['scale_factor'] * meta['obliquity_factor'] * meta['wavelength'] /
              ZYGO_PHASE_RES_FACTORS[meta['phase_res']]) * 1e9  # unit m to nm
    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }


def read_zygo_metadata(file_contents):
    """Parse metadata from the contents of a binary Zygo file.

    Parameters
    ----------
    file_contents : bytes
        binary file contents

    Returns
    -------
    dict
        dictionary with a shitload of keys for all of Zygo's metadata.

    """
    # convenient single character name
    c = file_contents
    IB16 = '>H'
    IL16 = '<H'
    IB32 = '>I'
    IL32 = '<I'
    FB32 = '>f'
    FL32 = '<f'
    C = 'c'
    uint8 = 'B'
    WASTE_BYTE = '\x00'

    magic_number = struct.unpack(IB32, c[:4])[0]
    header = {
        'format': struct.unpack(IB16, c[4:6])[0],
        'size': struct.unpack(IB32, c[6:10])[0],
    }
    swtype = struct.unpack(IB16, c[10:12])[0]
    swdate = c[12:42].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    swmaj = struct.unpack(IB16, c[42:44])[0]
    swmin = struct.unpack(IB16, c[44:46])[0]
    swpatch = struct.unpack(IB16, c[46:48])[0]
    metropro_version = f'{swmaj}.{swmin}.{swpatch}'
    ac = {
        'x': struct.unpack(IB16, c[48:50])[0],
        'y': struct.unpack(IB16, c[50:52])[0],
        'width': struct.unpack(IB16, c[52:54])[0],
        'height': struct.unpack(IB16, c[54:56])[0],
        'n_buckets': struct.unpack(IB16, c[56:58])[0],
        'range': struct.unpack(IB16, c[58:60])[0],
        'n_bytes': struct.unpack(IB32, c[60:64])[0],
    }
    cn = {
        'x': struct.unpack(IB16, c[64:66])[0],
        'y': struct.unpack(IB16, c[66:68])[0],
        'width': struct.unpack(IB16, c[68:70])[0],
        'height': struct.unpack(IB16, c[70:72])[0],
        'n_bytes': struct.unpack(IB32, c[72:76])[0],
    }
    timestamp = struct.unpack(IB32, c[76:80])[0]
    comment = c[80:162].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    source = struct.unpack(IB16, c[162:164])[0]
    scale_factor = struct.unpack(FB32, c[164:168])[0]
    wavelength = struct.unpack(FB32, c[168:172])[0]
    numerical_aperture = struct.unpack(FB32, c[172:176])[0]
    obliquity_factor = struct.unpack(FB32, c[176:180])[0]
    magnification = struct.unpack(FB32, c[180:184])[0]
    lateral_resolution = struct.unpack(FB32, c[184:188])[0]
    acq_type = struct.unpack(IB16, c[188:190])[0]
    intensity_average_count = struct.unpack(IB16, c[190:192])[0]
    sfac_limit = struct.unpack(IB16, c[194:196])[0]
    ramp = {
        'cal': struct.unpack(IB16, c[192:194])[0],
        'gain': struct.unpack(IB16, c[196:198])[0],
    }
    part_thickness = struct.unpack(FB32, c[198:202])[0]
    sw_llc = struct.unpack(IB16, c[202:204])[0]
    target_range = struct.unpack(FB32, c[204:208])[0]
    rad_crv_measure_seq = struct.unpack(IL16, c[208:210])[0]
    min_mod = struct.unpack(IB32, c[210:214])[0]
    min_mod_count = struct.unpack(IB32, c[214:218])[0]
    phase_res = struct.unpack(IB16, c[218:220])[0]
    min_area = struct.unpack(IB32, c[220:224])[0]
    discontinuity = {
        'action': struct.unpack(IB16, c[224:226])[0],
        'filter': struct.unpack(FB32, c[226:230])[0],
    }
    connect_order = struct.unpack(IB16, c[230:232])[0]
    sign = struct.unpack(IB16, c[232:234])[0]
    camera = {
        'width': struct.unpack(IB16, c[234:236])[0],
        'height': struct.unpack(IB16, c[236:238])[0],
    }
    _sys = {
        'type': struct.unpack(IB16, c[238:240])[0],
        'board': struct.unpack(IB16, c[240:242])[0],
        'serial': struct.unpack(IB16, c[242:244])[0],
        'inst_id': struct.unpack(IB16, c[244:246])[0]
    }
    obj_name = c[246:258].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    part_name = c[258:298].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    codev_type = struct.unpack(IB16, c[298:300])[0]
    phase_avg_count = struct.unpack(IB16, c[300:302])[0]
    sub_sys_err = struct.unpack(IB16, c[302:304])[0]
    # 305-320 unused
    part_sn = c[320:360].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    refractive_index = struct.unpack(FB32, c[360:364])[0]
    remove = {
        'tilt': struct.unpack(IB16, c[364:366])[0],
        'fringes': struct.unpack(IB16, c[366:368])[0],
    }
    max_area = struct.unpack(IB32, c[368:372])[0]
    setup_type = struct.unpack(IB16, c[372:374])[0]
    wrapped = struct.unpack(IB16, c[374:376])[0]
    pre_connect_filter = struct.unpack(FB32, c[376:380])[0]
    wavelength_in = {
        1: struct.unpack(FB32, c[386:390])[0],
        2: struct.unpack(FB32, c[380:384])[0],
        3: struct.unpack(FB32, c[390:394])[0],
        4: struct.unpack(FB32, c[394:398])[0],
        'fold': struct.unpack(IB16, c[384:386])[0],
    }
    wavelength_select = c[398:406].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    fda_res = struct.unpack(IB16, c[406:408])[0]
    scan_description = c[408:428].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    # n_fiducials = struct.unpack(IB16, c[428:430])  # skip - redundant
    fiducials = [
        struct.unpack(FB32, c[430:434])[0],
        struct.unpack(FB32, c[434:438])[0],
        struct.unpack(FB32, c[438:442])[0],
        struct.unpack(FB32, c[442:446])[0],
        struct.unpack(FB32, c[446:450])[0],
        struct.unpack(FB32, c[450:454])[0],
        struct.unpack(FB32, c[454:458])[0],
        struct.unpack(FB32, c[458:462])[0],
        struct.unpack(FB32, c[462:466])[0],
        struct.unpack(FB32, c[466:470])[0],
        struct.unpack(FB32, c[470:474])[0],
        struct.unpack(FB32, c[474:478])[0],
        struct.unpack(FB32, c[478:482])[0],
        struct.unpack(FB32, c[482:486])[0],
    ]
    pixel_dims = {
        'width': struct.unpack(FB32, c[486:490])[0],
        'height': struct.unpack(FB32, c[490:494])[0]
    }
    exit_pupil_diameter = struct.unpack(FB32, c[494:498])[0]
    light_level_percent = struct.unpack(FB32, c[498:502])[0]
    coords = {
        'state': struct.unpack(IL32, c[502:506])[0],
        'x': struct.unpack(FL32, c[506:510])[0],
        'y': struct.unpack(FL32, c[510:514])[0],
        'z': struct.unpack(FL32, c[514:518])[0],
        'a': struct.unpack(FL32, c[518:522])[0],  # x rotation
        'b': struct.unpack(FL32, c[522:526])[0],  # y rotation
        'c': struct.unpack(FL32, c[526:530])[0],  # z rotation
    }
    coherence_mode = struct.unpack(IL16, c[530:532])[0]
    surface_filter = struct.unpack(IL16, c[532:534])[0]
    sys_err_filename = c[534:562].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    zoom_descr = c[562:570].decode(ZYGO_ENC).rstrip(WASTE_BYTE)
    # maybe can make a part dict, merge with above part_thickness, etc?
    alpha_part = struct.unpack(FL32, c[570:574])[0]
    beta_part = struct.unpack(FL32, c[574:578])[0]
    dist_part = struct.unpack(FL32, c[578:582])[0]
    cam_split = {
        'loc_x': struct.unpack(IL16, c[582:584])[0],
        'loc_y': struct.unpack(IL16, c[584:586])[0],
        'trans_x': struct.unpack(IL16, c[586:588])[0],
        'trans_y': struct.unpack(IL16, c[588:590])[0],
    }
    material = {
        'a': c[590:614].decode(ZYGO_ENC).rstrip(WASTE_BYTE),
        'b': c[614:638].decode(ZYGO_ENC).rstrip(WASTE_BYTE),
    }
    # 639-642 unused
    dmi_center = {
        'x': struct.unpack(FL32, c[642:646])[0],
        'y': struct.unpack(FL32, c[646:650])[0],
    }
    sph_distortion_correction = struct.unpack(IL16, c[650:652])[0]
    # 653-654 unused
    sph_dist = {
        'part_na': struct.unpack(FL32, c[654:658])[0],
        'part_radius': struct.unpack(FL32, c[658:662])[0],
        'cal_na': struct.unpack(FL32, c[662:666])[0],
        'cal_radius': struct.unpack(FL32, c[666:670])[0],
    }
    surface_type = struct.unpack(IL16, c[670:672])[0]
    ac_surface_type = struct.unpack(IL16, c[672:674])[0]
    z_pos = struct.unpack(FL32, c[674:678])[0]
    power_mul = struct.unpack(FL32, c[678:682])[0]
    focus_mul = struct.unpack(FL32, c[682:686])[0]
    roc_focus_cal_factor = struct.unpack(FL32, c[686:690])[0]
    roc_power_cal_factor = struct.unpack(FL32, c[690:694])[0]
    ftp_pos = {
        'left': struct.unpack(FL32, c[694:698])[0],
        'right': struct.unpack(FL32, c[698:702])[0],
        'pitch': struct.unpack(FL32, c[702:706])[0],
        'roll': struct.unpack(FL32, c[706:710])[0],
    }
    min_mod_percent = struct.unpack(FL32, c[710:714])[0]
    max_intens = struct.unpack(IL32, c[714:718])[0]
    ring_of_fire = struct.unpack(IL16, c[718:720])[0]  # lol wyd zygo
    # 721 unused
    rc = {
        'orientation': struct.unpack(C, c[721:722])[0].decode(ZYGO_ENC).rstrip(WASTE_BYTE),
        'distance': struct.unpack(FL32, c[722:726])[0],
        'angle': struct.unpack(FL32, c[726:730])[0],
        'diameter': struct.unpack(FL32, c[730:734])[0],
    }
    rem_fringes_mode = struct.unpack(IB16, c[734:736])[0]
    # 737 unused
    ftpsi_phase_res = struct.unpack(uint8, c[737:738])[0]
    frames_acquired = struct.unpack(IL16, c[738:740])[0]
    cavity_type = struct.unpack(IL16, c[740:742])[0]
    cam_frame_rate = struct.unpack(FL32, c[742:746])[0]
    tune_range = struct.unpack(FL32, c[746:750])[0]
    cal_pix = {
        'x': struct.unpack(IL16, c[750:752])[0],
        'y': struct.unpack(IL16, c[752:754])[0],
    }
    # n_test_cal_pts = struct.unpack(IL16, c[754:756])  # not bothering to read
    # n_ref_cal_pts = struct.unpack(IL16, c[756:758])   # these, redundant
    test_cal_pts = [
        struct.unpack(FL32, c[758:762])[0],
        struct.unpack(FL32, c[762:766])[0],
        struct.unpack(FL32, c[766:770])[0],
        struct.unpack(FL32, c[770:774])[0],
    ]
    ref_cal_pts = [
        struct.unpack(FL32, c[774:778])[0],
        struct.unpack(FL32, c[778:782])[0],
        struct.unpack(FL32, c[782:786])[0],
        struct.unpack(FL32, c[786:790])[0],
    ]
    test_cal_pix_opd = struct.unpack(FL32, c[790:794])[0]
    test_ref_pix_opd = struct.unpack(FL32, c[794:798])[0]
    flash_phase_cd_mask = struct.unpack(FL32, c[798:802])[0]
    flash_phase_alias_mask = struct.unpack(FL32, c[802:806])[0]
    flask_phase_filter = struct.unpack(FL32, c[806:810])[0]
    scan_direction = struct.unpack(uint8, c[810:811])[0]
    # 812 - 814 unused
    ftpsi_res_factor = struct.unpack(IL16, c[814:816])[0]
    # 835 - 900 films, for later
    # 901 - 4096 unused

    # combine distant vars
    scan = {
        'direction': scan_direction,
        'scan_description': scan_description,
    }
    all_vars = [
        magic_number,
        header,
        swtype,
        swdate,
        metropro_version,
        ac,
        cn,
        timestamp,
        comment,
        source,
        scale_factor,
        wavelength,
        numerical_aperture,
        obliquity_factor,
        magnification,
        lateral_resolution,
        acq_type,
        intensity_average_count,
        ramp,
        sfac_limit,
        part_thickness,
        sw_llc,
        target_range,
        rad_crv_measure_seq,
        min_mod,
        min_mod_count,
        phase_res,
        min_area,
        discontinuity,
        connect_order,
        sign,
        camera,
        _sys,
        obj_name,
        part_name,
        codev_type,
        phase_avg_count,
        sub_sys_err,
        part_sn,
        refractive_index,
        remove,
        max_area,
        setup_type,
        wrapped,
        pre_connect_filter,
        wavelength_in,
        wavelength_select,
        fda_res,
        scan,
        fiducials,
        pixel_dims,
        exit_pupil_diameter,
        light_level_percent,
        coords,
        coherence_mode,
        surface_filter,
        sys_err_filename,
        zoom_descr,
        alpha_part,
        beta_part,
        dist_part,
        cam_split,
        material,
        dmi_center,
        sph_distortion_correction,
        sph_dist,
        surface_type,
        ac_surface_type,
        z_pos,
        power_mul,
        focus_mul,
        roc_focus_cal_factor,
        roc_power_cal_factor,
        ftp_pos,
        min_mod_percent,
        max_intens,
        ring_of_fire,
        rc,
        rem_fringes_mode,
        ftpsi_phase_res,
        frames_acquired,
        cavity_type,
        cam_frame_rate,
        tune_range,
        cal_pix,
        ref_cal_pts,
        test_cal_pts,
        test_cal_pix_opd,
        test_ref_pix_opd,
        flash_phase_cd_mask,
        flash_phase_alias_mask,
        flask_phase_filter,
        ftpsi_res_factor,
    ]
    all_keys = [
        'magic_number',
        'header',
        'swtype',
        'swdate',
        'metropro_version',
        'ac',
        'cn',
        'timestamp',
        'comment',
        'source',
        'scale_factor',
        'wavelength',
        'numerical_aperture',
        'obliquity_factor',
        'magnification',
        'lateral_resolution',
        'acq_type',
        'intensity_average_count',
        'ramp',
        'sfac_limit',
        'part_thickness',
        'sw_llc',
        'target_range',
        'rad_crv_measure_seq',
        'min_mod',
        'min_mod_count',
        'phase_res',
        'min_area',
        'discontinuity',
        'connect_order',
        'sign',
        'camera',
        '_sys',
        'obj_name',
        'part_name',
        'codev_type',
        'phase_avg_count',
        'sub_sys_err',
        'part_sn',
        'refractive_index',
        'remove',
        'max_area',
        'setup_type',
        'wrapped',
        'pre_connect_filter',
        'wavelength_in',
        'wavelength_select',
        'fda_res',
        'scan',
        'fiducials',
        'pixel_dims',
        'exit_pupil_diameter',
        'light_level_percent',
        'coords',
        'coherence_mode',
        'surface_filter',
        'sys_err_filename',
        'zoom_descr',
        'alpha_part',
        'beta_part',
        'dist_part',
        'cam_split',
        'material',
        'dmi_center',
        'sph_distortion_correction',
        'sph_dist',
        'surface_type',
        'ac_surface_type',
        'z_pos',
        'power_mul',
        'focus_mul',
        'roc_focus_cal_factor',
        'roc_power_cal_factor',
        'ftp_pos',
        'min_mod_percent',
        'max_intens',
        'ring_of_fire',
        'rc',
        'rem_fringes_mode',
        'ftpsi_phase_res',
        'frames_acquired',
        'cavity_type',
        'cam_frame_rate',
        'tune_range',
        'cal_pix',
        'ref_cal_pts',
        'test_cal_pts',
        'test_cal_pix_opd',
        'test_ref_pix_opd',
        'flash_phase_cd_mask',
        'flash_phase_alias_mask',
        'flask_phase_filter',
        'ftpsi_res_factor',
    ]

    return {k: v for k, v in zip(all_keys, all_vars)}


def write_zygo_ascii(file, phase, dx, wavelength=0.6328, intensity=None):
    """Write a Zygo ASCII interferogram file.

    Parameters
    ----------
    file : str
        filename
    phase : numpy.ndarray
        array of phase values
    dx : numpy.ndarray
        inter-sample spacing, mm
    wavelength : float, optional
        wavelength of light, um
    intensity : numpy.ndarray, optional
        intensity data

    """
    # construct the header
    timestamp = datetime.datetime.now()
    line1 = 'Zygo ASCII Data File - Format 2'
    line2 = '0 0 0 0 ' + timestamp.strftime('"%a %b %d %H:%M:%S %Y').ljust(30, ' ') + '"'
    if intensity is None:
        line3 = '0 0 0 0 0 0'
    else:
        raise NotImplementedError('writing of ASCII files with nonempty intensity not yet supported.')
    px, py = phase.shape
    ox = 0
    oy = 0
    line4 = f'{oy} {ox} {py} {px}'
    line5 = '"' + ' ' * 81 + '"'
    line6 = '"' + ' ' * 39 + '"'
    line7 = '"' + ' ' * 39 + '"'

    timestamp_int = int(str(timestamp.timestamp()).split('.')[0])
    res = dx * 1e3
    line8 = f'0 0.5 {wavelength*1e-6} 0 1 0 {res} {timestamp_int}'  # end is timestamp in integer seconds
    line9 = f'{py} {px} 0 0 0 0 ' + '"' + ' ' * 9 + '"'
    line10 = '0 0 0 0 0 0 0 0 0 0'
    line11 = '1 1 20 2 0 0 0 0 0'
    line12 = '0 ' + '"' + ' ' * 12 + '"'
    line13 = '1 0'
    line14 = '"' + ' ' * 7 + '"'

    header_lines = (line1,
                    line2,
                    line3,
                    line4,
                    line5,
                    line6,
                    line7,
                    line8,
                    line9,
                    line10,
                    line11,
                    line12,
                    line13,
                    line14)
    header = '\n'.join(header_lines) + '\n'

    if intensity is None:
        line15 = '#'

    line16 = '#'

    # process the phase and write out
    coef = ZYGO_PHASE_RES_FACTORS[1]
    encoded_phase = phase * (coef / wavelength / wavelength / 0.5)
    encoded_phase[np.isnan(encoded_phase)] = ZYGO_INVALID_PHASE
    encoded_phase = encoded_phase.astype(np.int64)
    encoded_phase = encoded_phase.flatten()
    npts = encoded_phase.shape[0]
    fits_by_ten = npts // 10
    boundary = 10 * fits_by_ten

    # create an in-memory buffer and write out the phase to it
    s = StringIO()
    s.write(header)
    s.write('\n'.join([line15, line16, '']))
    truenp.savetxt(s, encoded_phase[:boundary].reshape(-1, 10), fmt='%d', delimiter=' ', newline=' \n')
    tail = ' '.join((str(d) for d in encoded_phase[boundary:]))
    s.write(tail)
    s.write('\n#\n')
    s.seek(0)

    if not isinstance(file, IOBase):
        with open(file, 'w') as fd:
            shutil.copyfileobj(s, fd)
    else:
        shutil.copyfileobj(s, file)


def read_sigfit_zernikes(file):
    """Read Zernikes from a SigFit OUTCOF3 file.

    Parameters
    ----------
    file : str or Path_like
        path to a file

    Returns
    -------
    dict with keys of surface IDs, which have values of dicts with keys of:
        - type | Noll ("Zemax Standard") or Fringe Zernikes
        - normed | if True, the terms are orthonormalized and have unit standard deviation, else unit amplitude
        - wavelength | wavelength of light in microns
        - rnorm | normalization radius, mm
        - coefs | Zernike mode coefficients, microns

    """
    with open(str(file), 'r') as fid:
        data = fid.read()

    data = data.split('Surface')
    out = {}
    for dat in data[1:]:
        sid, data = _read_sigfit_zernike_core(dat)
        out[sid] = data
    return out


def _read_sigfit_zernike_core(text):
    lines = text.splitlines()
    _, rest = lines[0].split('SID=')
    rest = rest.lstrip()
    surface, rest = rest.split('Rnorm=')
    surface = int(surface.strip())
    rest = rest.lstrip()
    rnorm, rest = rest.split('Type')
    _, rest = rest.split('WVL=')
    rest = rest.lstrip()
    wvl, rest = rest.split()
    unit = rest.lstrip()
    fctr = 25.4e3 if unit.lower() == 'in' else 1e3
    typ = 'Noll' if 'ZEMAX' in lines[2] else 'Fringe'
    normed = True if 'RMS' in lines[2] else False
    rnorm = float(rnorm.lstrip()) * fctr / 1e3
    coefs = []
    total_lines = len(lines)
    if lines[-1].strip() == '':
        slice_ = slice(4, -1)
    else:
        slice_ = slice(4, total_lines - 1)
    for line in lines[slice_]:  # last line is blank
        idx, *coef = line.split(',')
        if isinstance(coef, list) and len(coef) > 0:
            coef, *_ = coef

        if coef == '' or coef == []:
            coefs.append(0)
        else:
            coefs.append(float(coef))

    coefs = np.asarray(coefs)

    wvl = float(wvl) * fctr
    return surface, {
        'type': typ,
        'normed': normed,
        'wavelength': wvl,
        'coefs': coefs * wvl,
        'rnorm': rnorm,
    }


def read_sigfit_rigidbody(file):
    """Read rigid body perturbation data from a SigFit sum1.csv file.

    Parameters
    ----------
    file : str or path_like
        location of a sigfit sum1.csv file

    Returns
    -------
    dict with keys of surface IDs, which have values of dicts with keys of dx, dy, dz, rx, ry, rz, dR
        all values in mm

    """
    file = str(file)
    with open(file, 'r') as fid:
        data = fid.readlines()

    if '= in' in data[4]:
        fctr = 25.4
    else:
        fctr = 1

    data = truenp.genfromtxt(file, skip_header=7, delimiter=',')[:, 4:12]
    data[:, 1:] *= fctr
    out = {}
    for row in data:
        sid, dx, dy, dz, rx, ry, rz, dR = row
        out[int(sid)] = {
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'rx': rx,
            'ry': ry,
            'rz': rz,
            'dR': dR
        }
    return out


def _find_nth(string, substring, n):
    start = string.find(substring)
    l = len(substring)  # NOQA
    while start >= 0 and n > 1:
        start = string.find(substring, start+l)
        n -= 1
    return start


def read_codev_gridint(file):
    """Read a Code V INT file containing grid data.

    Parameters
    ----------
    file : str or path_like
        path to a grid int file

    Returns
    -------
    tuple of (ndarray, dict)
        grid data in array representation, metadata dict

    """
    txt = Path(file).expanduser().read_text()
    # feed-forward information that prevents us from doing a whole-text search:
    # the manual specifies that each record must be <= 80 characters, so we
    # can look at 80 character chunks and test for apostrophies
    # this will break for microscopic int files, say 8x8.  I accept the bug
    end = 80
    while True:
        l = len(txt)  # NOQA - l short
        if l < end:
            end = l
        # it may strictly speaking be faster to compare txt[0] to !, but oh well
        i = txt[:end].find('!')
        if i < 0:  # no more comments
            break

        # we are in a comment, find the newline and skip over that line
        i = txt.find('\n', i)  # starting from i is a very mild performance improvement
        if i < 0:
            raise ValueError('CV INT file header corrupted - no new line found after !')
        # skip forward
        txt = txt[i+1:]

    # now on the title line, look for the newline
    end = txt.find('\n')
    if end < 0:
        raise ValueError('CV INT file header corrupted - no new line found after title')

    title = txt[:end]

    # now on the header line, split that off
    txt = txt[end+1:]
    end = txt.find('\n')
    hdr = txt[:end]

    # parsing the header,
    # it is made up of Code V three-letter acronyms and their values
    # a limited parser here of the ones we know how to deal with
    params = hdr.split()  # some tokens are specifiers while others are values
    i = 0
    l = len(params)  # NOQA
    wvl, nda = None, None
    while i < l:
        if params[i].upper() == 'WVL':
            wvl = float(params[i+1])  # Code V uses microns for this unit, OK
            i += 2
            continue
        if params[i].upper() == 'SSZ':
            ssz = float(params[i+1])  # integers per wavelength of OPD/surface deformation
            i += 2
            continue
        if params[i].upper() == 'NDA':
            nda = int(params[i+1])
            i += 2
            continue
        if params[i].upper() == 'GRD':
            m = int(params[i+1])
            n = int(params[i+2])
            i += 3
            continue
        if params[i].upper() == 'SUR':
            meaning = 'surface error'
            i += 1
            continue
        if params[i].upper() == 'WFR':
            meaning = 'wavefront error'
            i += 1
            continue

        raise ValueError(f'parsing CV INT header: token {params[i]} not understood')

    if wvl is None:
        raise ValueError('CV INT header did not contain WVL')

    if nda is None:
        raise ValueError('CV INT (GRID) header did not contain NDA')

    if m is None or n is None:
        raise ValueError('CV INT header did not contain GRD, only grid INT files are supported')

    main_data = txt[end+1:]
    a = np.fromstring(main_data, sep=' ', dtype=np.int64)
    mask = a == nda
    # div by ssz converts to wvl, div by wvl to um, *1000 to nm
    a = a.astype(config.precision) * (1000/wvl/ssz)
    a[mask] = np.nan
    a = a.reshape((m, n))
    meta = {
        'title': title,
        'wavelength': wvl,
        'data meaning': meaning,
    }
    return a, meta
