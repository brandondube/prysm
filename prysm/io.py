"""File readers for various commercial instruments."""
import os
import re
import codecs
import datetime
import calendar

from .conf import config
from prysm import mathops as m


def read_file_stream_or_path(path_or_file):
    try:
        with codecs.open(path_or_file, mode='r', encoding='cp1252') as fid:
            data = codecs.encode(fid.read(), 'utf-8').decode('utf-8')
    except FileNotFoundError:
        path_or_file.seek(0)
        data = path_or_file.read()
    except AttributeError:
        data = path_or_file

    return data


def read_trioptics_mtfvfvf(file_path):
    """Read MTF vs Field vs Focus data from a Trioptics .txt dump.

    Parameters
    ----------
    file_path : path_like or `str`
        path to read data from

    Returns
    -------
    `MTFvFvF`
        MTF vs Field vs Focus object

    """
    with open(file_path, 'r') as fid:
        lines = fid.readlines()
        imghts, objangs, focusposes, mtfs = [], [], [], []
        for meta, data in zip(lines[0::2], lines[1::2]):  # iterate 2 lines at a time
            metavalues = meta.split()
            imght, objang, focuspos, freqpitch = metavalues[1::2]
            mtf_raw = data.split()[1:]  # first element is "MTF"
            mtf = m.asarray(mtf_raw, dtype=config.precision)
            imghts.append(imght)
            objangs.append(objang)
            focusposes.append(focuspos)
            mtfs.append(mtf)

    if str(file_path)[-7:-4] == 'Tan':
        azimuth = 'Tan'
    else:
        azimuth = 'Sag'

    focuses = m.unique(m.asarray(focusposes, dtype=config.precision))
    focuses = (focuses - m.mean(focuses)) * 1e3
    imghts = m.unique(m.asarray(imghts, dtype=config.precision))
    freqs = m.arange(len(mtfs[0]), dtype=config.precision) * float(freqpitch)
    data = m.swapaxes(m.asarray(mtfs).reshape(len(focuses), len(imghts), len(freqs)), 0, 1)
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
    file : `str` or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : `bool`
        whether to also extract and return metadata

    Returns
    -------
    `dict`
        dictionary with keys of freq, field, tan, sag

    """
    data = read_file_stream_or_path(file)
    data = data[:len(data)//10]  # only search in a subset of the file for speed

    # compile a pattern that will search for the image heights in the file and extract
    fields_pattern = re.compile(f'MTF=09{os.linesep}(.*?){os.linesep}Legend=09', flags=re.DOTALL)
    fields = fields_pattern.findall(data)[0]  # two copies, only need 1st

    # make a pattern that will search for and extract the tan and sag MTF data.  The match will
    # return two copies; one for vs imght, one for vs angle.  Only keep half the matches.
    tan_pattern = re.compile(r'Tan(.*?)=97', flags=re.DOTALL)
    sag_pattern = re.compile(r'Sag(.*?)=97', flags=re.DOTALL)
    tan, sag = tan_pattern.findall(data), sag_pattern.findall(data)
    endpt = len(tan) // 2
    tan, sag = tan[:endpt], sag[:endpt]

    # now extract the freqs from the tan data
    freqs = m.asarray([float(s.split('(')[0][1:]) for s in tan])

    # lastly, extract the floating point tan and sag data
    # also take fields, to the 4th decimal place (nearest .1um)
    # reformat T/S to 2D arrays with indices of (freq, field)
    tan = m.asarray([s.split('=09')[1:-1] for s in tan], dtype=config.precision)
    sag = m.asarray([s.split('=09')[1:-1] for s in sag], dtype=config.precision)
    fields = m.asarray(fields.split('=09')[0:-1], dtype=config.precision).round(4)
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


def read_trioptics_mtf(file, metadata=False):
    """Read MTF data from a Trioptics data file.

    Parameters
    ----------
    file : `str` or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : `bool`
        whether to also extract and return metadata

    Returns
    -------
    `dict`
        dictionary with keys focus, wavelength, freq, tan, sag
        if metadata=True, also has keys in the return of
        `io.parse_trioptics_metadata`.

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
    t = m.asarray(mtfs[:breakpt], dtype=config.precision)
    s = m.asarray(mtfs[breakpt:], dtype=config.precision)
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
    month_num = list(calendar.month_name).index(month)
    h, m, s = hms.split(':')
    timestamp = datetime.datetime(year=year, month=month_num, day=day, hour=h, minute=m, second=s)
    sampleid = sampleid_scanner.search(data).group(1)
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
        'instrument_sn': instrument_sn,
        'collimator': collimator_efl,
        'wavelength': wavelength,
        'efl': sample_efl,
        'obj_angle': obj_angle,
        'focus_pos': focus_pos,
        'azimuth': azimuth,
    }


def identify_trioptics_measurement_type(file):
    data = read_file_stream_or_path(file)
    data_parse = data[750:1500]
    measurement_type_scanner = re.compile(r'Measure Program  : (.*)')
    program = measurement_type_scanner.search(data_parse).group(1)
    return program, data


TRIOPTICS_SWITCHBOARD = {
    'MTF vs. Field': read_trioptics_mtf_vs_field,
    'Distortion': NotImplemented,
    'Axial Color': NotImplemented,
    'Lateral Color': NotImplemented,
}


def read_any_trioptics_mht(file):
    type_, data = identify_trioptics_measurement_type(file)
    return TRIOPTICS_SWITCHBOARD[type_](data)
