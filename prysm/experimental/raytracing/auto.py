"""Automatic design routines."""


def rc_prescription_from_efl_bfl_sep(efl, bfl, sep):
    """Design a Ritchey-Chrétien telescope from its spacings.

    Parameters
    ----------
    efl : float
        system effective focal length
    bfl : float
        system back focal length, distance from SM vertex to image
    sep : float
        the vertex-to-vertex separation of the two mirrors.

    Returns
    -------
    float, float, float, float
        c1, c2, k1, k2

    """
    F = efl
    B = bfl
    D = sep
    M = (F-B)/D  # secondary mirror magnification
    R1 = -(2*D*F)/(F-B)
    R2 = -(2*D*B)/(F-B-D)

    k1 = -1 - 2/M**3 * B/D
    k2 = -1 - 2/(M-1)**3 * (M*(2*M-1) + B/D)
    c1 = 1/R1
    c2 = 1/R2
    return c1, c2, k1, k2


def rc_prescription_from_pm_and_imc(efl, f_pm, pm_vertex_to_focus):
    """Design a Ritchey-Chrétien telescope from information about its primary mirror.

    Parameters
    ----------
    efl : float
        system effective focal length
    f_pm : float
        focal length of the primary mirror (should be negative)
    pm_vertex_to_focus : float
        the distance from the primary mirror vertex to the focus
        remember that the pm has nonzero thickness

    Returns
    -------
    float, float, float, float
        c1, c2, k1, k2

    """
    b = pm_vertex_to_focus
    F = efl
    D = f_pm * (F-b)
    B = D + b
    M = (F-B)/D  # secondary mirror magnification
    R1 = -(2*D*F)/(F-B)
    R2 = -(2*D*B)/(F-B-D)

    k1 = -1 - 2/M**3 * B/D
    k2 = -1 - 2/(M-1)**3 * (M*(2*M-1) + B/D)
    c1 = 1/R1
    c2 = 1/R2
    return c1, c2, k1, k2
