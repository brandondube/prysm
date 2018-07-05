"""File readers for various commercial instruments."""
import os
import re
import struct
import codecs
import datetime
import calendar

from .conf import config
from prysm import mathops as m


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
    file : `str` or path_like or file_like
        file to read from, if string of file body, must provide filename

    Returns:
    boolean : `bool`
        if the file is an MTFvFvF file
    data : `str`
        contents of the file
    `
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
    file : `str` or path_like or file_like
        file to read from, if string of file body, must provide filename
    filename : `str`, optional
        name of file; used to select tan/sag if file is given as contents

    Returns
    -------
    `MTFvFvF`
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
        mtf = m.asarray(mtf_raw, dtype=config.precision)
        imghts.append(imght)
        objangs.append(objang)
        focusposes.append(focuspos)
        mtfs.append(mtf)

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
    """Read metadata from the contents of a Trioptics .mht file.

    Parameters:
    -----------
    file_contents : `str`
        contents of a .mht file.

    Returns:
    --------
    `dict`
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


def identify_trioptics_measurement_type(file):
    """Identify type of measurement in a Trioptics .mht file.

    Parameters
    ----------
    file : `str` or path_like or file_like
        contents of a file, path_like to the file, or file object

    Returns
    -------
    program : `str`
        measurement type
    data : `str`
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
    file : `str` or path_like or file_like
        contents of a file, path_like to the file, or file object
    metadata : `bool`
        whether to also extract and return metadata

    Returns
    -------
    `dict`
        dictionary with appropriate keys.  If metadata=True, also has keys in
        the return of `io.parse_trioptics_metadata`.

    """
    type_, data = identify_trioptics_measurement_type(file)
    return type_, TRIOPTICS_SWITCHBOARD[type_](data, metadata=metadata)

ZYGO_ENC = 'utf-8'  # may be ASCII, cp1252...
ZYGO_PHASE_RES_FACTORS = {
    0: 4096,
    1: 32768,
}

def read_zygo_dat(file):
    """Read the contents of a zygo binary (.dat) file.

    Parameters
    ----------
    file : path_like
        path to a file

    Returns
    -------
    `dict`
        dictionary with keys: phase, intensity, meta

    """
    with open(file, 'rb') as fid:
        contents = fid.read()

    meta = read_zygo_metadata(contents)
    w, h = meta['camera']['width'], meta['camera']['height']
    img_len = w * h
    header_len = meta['header']['size']
    intensity = m.frombuffer(contents, offset=header_len, count=img_len*2, dtype=m.uint16)
    intensity[intensity >= 2 ** 16 - 1] = m.nan
    phase = m.frombuffer(contents, offset=header_len+img_len, count=img_len*4, dtype=m.int32).astype(config.precision)
    phase[phase >= 2 ** 31 - 1] = m.nan
    phase *= (meta['scale_factor'] * meta['obliquity_factor'] * meta['wavelength'] /
              ZYGO_PHASE_RES_FACTORS[meta['phase_res']])
    return {
        'phase': phase,
        'intensity': intensity,
        'meta': meta,
    }


def read_zygo_metadata(file_contents):
    """Parse metadata from the contents of a binary Zygo file.

    Parameters
    ----------
    file_contents : `bytes`
        binary file contents

    Returns
    -------
    `dict`
        dictionary with a shitload of keys for all of Zygo's metadata.

    """
    # convenient single character name
    c = file_contents
    IB16 = '>H'
    IL16 = '<H'
    IB32 = '>I'
    IL32 = '<I'
    FB32 = '>f'
    FB64 = '>d'
    FL32 = '<f'
    FL64 = '<d'
    C = 'c'
    uint8 = 'B'

    magic_number = struct.unpack(IB32, c[:4])
    header = {
        'format': struct.unpack(IB16, c[4:6]),
        'size': struct.unpack(IB32, c[6:10]),
    }
    swmaj = struct.unpack(IB16, c[10:12])
    swmin = struct.unpack(IB16, c[12:14])
    swpatch = struct.unpack(IB16, c[14:16])
    metropro_version = f'{swmaj}.{swmin}.{swpatch}'
    ac = {
        'x': struct.unpack(IB16, c[48:50]),
        'y': struct.unpack(IB16, c[50:52]),
        'width': struct.unpack(IB16, c[52:54]),
        'height': struct.unpack(IB16, c[54:56]),
        'n_buckets': struct.unpack(IB16, c[56:58]),
        'range': struct.unpack(IB16, c[58:60]),
        'n_bytes': struct.unpack(IB32, c[60:64]),
    }
    cn = {
        'x': struct.unpack(IB16, c[64:66]),
        'y': struct.unpack(IB16, c[66:68]),
        'width': struct.unpack(IB16, c[68:70]),
        'height': struct.unpack(IB16, c[70:72]),
        'n_bytes': struct.unpack(IB32, c[72:76]),
    }
    timestamp = struct.unpack(IB32, c[76:80])
    comment = c[80:162].decode(ZYGO_ENC)
    source = struct.unpack(IB16, c[162:164])
    scale_factor = struct.unpack(FB32, c[164:168])
    wavelength = struct.unpack(FB32, c[168:172])
    numerical_aperture = struct.unpack(FB32, c[172:176])
    obliquity_factor = struct.unpack(FB32, c[176:180])
    magnification = struct.unpack(FB32, c[180:184])
    lateral_resolution = struct.unpack(FB32, c[184:188])
    acq_type = struct.unpack(IB16, c[188:190])
    intensity_average_count = struct.unpack(IB16, c[190:192])
    sfac_limit = struct.unpack(IB16, c[194:196])
    ramp = {
        'cal': struct.unpack(IB16, c[192:194]),
        'gain': struct.unpack(IB16, c[196:198]),
    }
    part_thickness = struct.unpack(FB32, c[198:202])
    sw_llc = struct.unpack(IB16, c[202:204])
    target_range = struct.unpack(FB32, c[204:208])
    rad_crv_measure_seq = struct.unpack(IL32, c[208:210])
    min_mod = struct.unpack(IB32, c[210:214])
    min_mod_count = struct.unpack(IB32, c[214:218])
    phase_res = struct.unpack(IB16, c[218:220])
    min_area = struct.unpack(IB32, c[220:224])
    discontinuity = {
        'action': struct.unpack(IB16, c[224:226]),
        'filter': struct.unpack(FB32, c[226:230]),
    }
    connect_order = struct.unpack(IB16, c[230:232])
    sign = struct.unpack(IB16, c[232:234])
    camera = {
        'width': struct.unpack(IB16, c[234:236]),
        'height': struct.unpack(IB16, c[236:238]),
    }
    _sys = {
        'type': struct.unpack(IB16, c[238:240]),
        'board': struct.unpack(IB16, c[240:242]),
        'serial': struct.unpack(IB16, c[242:244]),
        'inst_id': struct.unpack(IB16, c[244:246])
    }
    obj_name = c[246:258].decode(ZYGO_ENC)
    part_name = c[258:298].decode(ZYGO_ENC)
    codev_type = struct.unpack(IB16, c[298:300])
    phase_avg_count = struct.unpack(IB16, c[300:302])
    sub_sys_err = struct.unpack(IB16, c[302:304])
    # 305-320 unused
    part_sn = c[320:360].decode(ZYGO_ENC)
    refractive_index = struct.unpack(FB32, c[360:364])
    remove = {
        'tilt': struct.unpack(IB16, c[364:366]),
        'fringes': struct.unpack(IB16, c[366:368]),
    }
    max_area = struct.unpack(IB32, c[368:372])
    setup_type = struct.unpack(IB16, c[372:374])
    wrapped = struct.unpack(IB16, c[374:376])
    pre_connect_filter = struct.unpack(FB32, c[376:380])
    wavelength_in = {
        1: struct.unpack(FB32, c[386:390]),
        2: struct.unpack(FB32, c[380:384]),
        3: struct.unpack(FB32, c[390:394]),
        4: struct.unpack(FB32, c[394:398]),
        'fold': struct.unpack(IB16, c[386:390]),
    }
    wavelength_select = c[398:406].decode(ZYGO_ENC)
    fda_res = struct.unpack(IB16, c[406:408])
    scan_description = c[408:428].decode(ZYGO_ENC)
    # n_fiducials = struct.unpack(IB16, c[428:430])  # skip - redundant
    fiducials = [
        struct.unpack(FB32, c[430:434]),
        struct.unpack(FB32, c[434:438]),
        struct.unpack(FB32, c[438:442]),
        struct.unpack(FB32, c[442:446]),
        struct.unpack(FB32, c[446:450]),
        struct.unpack(FB32, c[450:454]),
        struct.unpack(FB32, c[454:458]),
        struct.unpack(FB32, c[458:462]),
        struct.unpack(FB32, c[462:466]),
        struct.unpack(FB32, c[466:470]),
        struct.unpack(FB32, c[470:474]),
        struct.unpack(FB32, c[474:478]),
        struct.unpack(FB32, c[478:482]),
        struct.unpack(FB32, c[482:486]),
    ]
    pixel_dims = {
        'width': struct.unpack(FB32, c[486:490]),
        'height': struct.unpack(FB32, c[490:494])
    }
    exit_pupil_diameter = struct.unpack(FB32, c[494:498])
    light_level_percent = struct.unpack(FB32, c[498:502])
    coords = {
        'state': struct.unpack(IL32, c[502:506]),
        'x': struct.unpack(FL32, c[506:510]),
        'y': struct.unpack(FL32, c[510:514]),
        'z': struct.unpack(FL32, c[514:518]),
        'a': struct.unpack(FL32, c[518:522]),  # x rotation
        'b': struct.unpack(FL32, c[522:526]),  # y rotation
        'c': struct.unpack(FL32, c[526:530]),  # z rotation
    }
    coherence_mode = struct.unpack(IL16, c[530:532])
    surface_filter = struct.unpack(IL16, c[532:534])
    sys_err_filename = c[534:562].decode(ZYGO_ENC)
    zoom_descr = c[562:570].decode(ZYGO_ENC)
    # maybe can make a part dict, merge with above part_thickness, etc?
    alpha_part = struct.unpack(FL32, c[570:574])
    beta_part = struct.unpack(FL32, c[574:578])
    dist_part = struct.unpack(FL32, c[578:582])
    cam_split = {
        'loc_x': struct.unpack(IL16, c[582:584]),
        'loc_y': struct.unpack(IL16, c[584:586]),
        'trans_x': struct.unpack(IL16, c[586:588]),
        'trans_y': struct.unpack(IL16, c[588:590]),
    }
    material = {
        'a': c[590:614].decode(ZYGO_ENC),
        'b': c[614:638].decode(ZYGO_ENC)
    }
    # 639-642 unused
    dmi_center = {
        'x': struct.unpack(FL32, c[642:646]),
        'y': struct.unpack(FL32, c[646:650]),
    }
    sph_distortion_correction = struct.unpack(IL16, c[650:652])
    # 653-654 unused
    sph_dist = {
        'part_na': struct.unpack(FL32, c[654:658]),
        'part_radius': struct.unpack(FL32, c[658:662]),
        'cal_na': struct.unpack(FL32, c[662:666]),
        'cal_radius': struct.unpack(FL32, c[666:670]),
    }
    surface_type = struct.unpack(IL16, c[670:672])
    ac_surface_type = struct.unpack(IL16, c[672:674])
    z_pos = struct.unpack(FL32, c[674:678])
    power_mul = struct.unpack(FL32, c[678:682])
    focus_mul = struct.unpack(FL32, c[682:686])
    roc_focus_cal_factor = struct.unpack(FL32, c[686:690])
    roc_power_cal_factor = struct.unpack(FL32, c[690:694])
    ftp_pos = {
        'left': struct.unpack(FL32, c[694:698]),
        'right': struct.unpack(FL32, c[698:702]),
        'pitch': struct.unpack(FL32, c[702:706]),
        'roll': struct.unpack(FL32, c[706:710]),
    }
    min_mod_percent = struct.unpack(FL32, c[710:714])
    max_intens = struct.unpack(IL32, c[714:718])
    ring_of_fire = struct.unpack(IL16, c[718:720])  # lol wyd zygo
    # 721 unused
    rc = {
        'orientation': struct.unpack(C, c[721:722]),
        'distance': struct.unpack(FL32, c[722:726]),
        'angle': struct.unpack(FL32, c[726:730]),
        'diameter': struct.unpack(FL32, c[730:734]),
    }
    rem_fringes_mode = struct.unpack(IB16, c[734:736])
    # 737 unused
    ftpsi_phase_res = struct.unpack(IL16, c[737:738])
    frames_acquired = struct.unpack(IL16, c[738:740])
    cavity_type = struct.unpack(IL16, c[740:742])
    cam_frame_rate = struct.unpack(FL32, c[742:746])
    tune_range = struct.unpack(FL32, c[746:750])
    cal_pix = {
        'x': struct.unpack(IL16, c[750:752]),
        'y': struct.unpack(IL16, c[752:754]),
    }
    # n_test_cal_pts = struct.unpack(IL16, c[754:756])  # not bothering to read
    # n_ref_cal_pts = struct.unpack(IL16, c[756:758])   # these, redundant
    test_cal_pts = [
        struct.unpack(FL32, c[758:762]),
        struct.unpack(FL32, c[762:766]),
        struct.unpack(FL32, c[766:770]),
        struct.unpack(FL32, c[770:774]),
    ]
    ref_cal_pts = [
        struct.unpack(FL32, c[774:778]),
        struct.unpack(FL32, c[778:782]),
        struct.unpack(FL32, c[782:786]),
        struct.unpack(FL32, c[786:790]),
    ]
    test_cal_pix_opd = struct.unpack(FL32, c[790:794])
    test_ref_pix_opd = struct.unpack(FL32, c[794:798])
    flash_phase_cd_mask = struct.unpack(FL32, c[798:802])
    flash_phase_alias_mask = struct.unpack(FL32, c[802:806])
    scan_direction = struct.unpack(uint8, c[806:806])
    # 812 - 814 unused
    ftpsi_res_factor = struct.unpack(IL16, c[814:816])
    # 835 - 900 films, for later
    # 901 - 4096 unused

    # combine distant vars
    scan = {
        'direction': scan_direction,
        'scan_description': scan_description,
    }
    all_vars = [magic_number, header, metropro_version, ac, cn, timestamp, comment,
        source, scale_factor, wavelength, numerical_aperture, obliquity_factor, magnification,
        lateral_resolution, acq_type, intensity_average_count, ramp, sfac_limit, part_thickness,
        sw_llc, target_range, rad_crv_measure_seq, min_mod, min_mod_count, phase_res, min_area,
        discontinuity, connect_order, sign, camera, _sys, obj_name, part_name, codev_type,
        phase_avg_count, sub_sys_err, part_sn, refractive_index, remove, max_area, setup_type,
        wrapped, pre_connect_filter, wavelength_in, wavelength_select, fda_res, scan, fiducials,
        pixel_dims, exit_pupil_diameter, light_level_percent, coords, coherence_mode, surface_filter,
        sys_err_filename, zoom_descr, alpha_part, beta_part, dist_part, cam_split, material,
        dmi_center, sph_distortion_correction, sph_dist, surface_type, ac_surface_type, z_pos,
        power_mul, focus_mul, roc_focus_cal_factor, roc_power_cal_factor, ftp_pos,
        min_mod_percent, max_intens, ring_of_fire, rc, rem_fringes_mode, ftpsi_phase_res,
        frames_acquired, cavity_type, cam_frame_rate, tune_range, cal_pix,
        ref_cal_pts, test_cal_pts, test_cal_pix_opd, test_ref_pix_opd, flash_phase_cd_mask,
        ftpsi_res_factor]

    return {var:var for var in all_vars}
