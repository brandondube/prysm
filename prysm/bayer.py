from .mathops import np, ndimage

top_left = (slice(0, None, 2), slice(0, None, 2))
top_right = (slice(1, None, 2), slice(0, None, 2))
bottom_left = (slice(0, None, 2), slice(1, None, 2))
bottom_right = (slice(1, None, 2), slice(1, None, 2))

ErrBadCFA = NotImplementedError('only rggb, bggr bayer patterns currently implemented')


def composite_bayer(r, g1, g2, b, cfa='rggb', output=None):
    """Composite an interleaved image from densely sampled bayer color planes.

    Parameters
    ----------
    r : `numpy.ndarray`
        ndarray of shape (m, n)
    g1 : `numpy.ndarray`
        ndarray of shape (m, n)
    g2 : `numpy.ndarray`
        ndarray of shape (m, n)
    b : `numpy.ndarray`
        ndarray of shape (m, n)
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arangement
    output : `numpy.ndarray`, optional
        output array, of shape (m, n) and same dtype as r, g1, g2, b

    Returns
    -------
    `numpy.ndarray`
        array of interleaved data

    """
    if output is None:
        output = np.empty_like(r)

    cfa = cfa.lower()
    if cfa == 'rggb':
        output[top_left] = r[top_left]
        output[top_right] = g1[top_right]
        output[bottom_left] = g2[bottom_left]
        output[bottom_right] = b[bottom_right]
    elif cfa == 'bggr':
        output[top_left] = b[top_left]
        output[top_right] = g1[top_right]
        output[bottom_left] = g2[bottom_left]
        output[bottom_right] = r[bottom_right]
    else:
        raise ErrBadCFA

    return output


# Kernels from Malvar et al, fig 2.
# names derived from the paper,
# in demosaic_malvar the naming
# may be more clear
# "G at R locations" or G at B locations
kernel_G_at_R_or_B = [
    [ 0, 0, -1, 0,  0], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [-1, 2,  4, 2, -1], # NOQA
    [ 0, 0,  2, 0,  0], # NOQA
    [ 0, 0, -1, 0,  0], # NOQA
]

# R at green in R row, B column
kernel_R_at_G_in_RB = [
    [ 0,  0, .5, 0,  0], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [-1,  4, 5,  4, -1], # NOQA
    [ 0, -1, 0, -1,  0], # NOQA
    [ 0,  0, .5, 0,  0], # NOQA
]

kernel_R_at_G_in_BR = [
    [0,  0, -1,  0, 0 ], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [.5, 0,  5,  0, .5], # NOQA
    [0, -1,  4, -1, 0 ], # NOQA
    [0,  0, -1,  0, 0 ], # NOQA
]

kernel_R_at_B_in_BB = [
    [0,    0, -3/2, 0,  0],   # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [-3/2, 0,  6,   0, -3/2], # NOQA
    [0,    2,  0,   2,  0],   # NOQA
    [0,    0, -3/2, 0,  0],   # NOQA
]


kernel_B_at_G_BR = kernel_R_at_G_in_RB
kernel_B_at_G_RB = kernel_R_at_G_in_BR
kernel_B_at_R_in_RR = kernel_R_at_B_in_BB


def demosaic_malvar(img, cfa='rggb'):
    """Demosaic an image using the Malvar algorithm.

    Parameters
    ----------
    img : `numpy.ndarray`
        ndarray of shape (m, n) containing mosaiced (interleaved) pixel data,
        as from a raw file
    cfa : `str`, optional, {'rggb', 'bggr'}
        color filter arrangement

    Returns
    -------
    `numpy.ndarray`
        ndarray of shape (m, n, 3) that has been demosaiced.  Final dimension
        is ordered R, G, B.  Is of the same dtype as img and has the same energy
        content and sense of z scaling

    """
    cfa = cfa.lower()
    # create all of our convolution kernels (FIR filters)
    # division by 8 is to make the kernel sum to 1
    # (preserve energy)
    kgreen = np.array(kernel_G_at_R_or_B) / 8
    kgreensameColumn = np.array(kernel_R_at_G_in_RB) / 8
    kgreensameRow = np.array(kernel_R_at_G_in_BR) / 8
    kdiagonalRB = np.array(kernel_R_at_B_in_BB) / 8

    # there is only one filter for G
    Gest = ndimage.convolve(img, kgreen)

    # there are only three unique convolutions remaining
    c1 = ndimage.convolve(img, kgreensameColumn)
    c2 = ndimage.convolve(img, kgreensameRow)
    c3 = ndimage.convolve(img, kdiagonalRB)

    red = np.empty_like(img)
    green = Gest
    blue = np.empty_like(img)

    green[top_right] = img[top_right]
    green[bottom_left] = img[bottom_left]

    if cfa == 'rggb':
        red[top_left] = img[top_left]
        red[top_right] = c2[top_right]
        red[bottom_left] = c1[bottom_left]
        red[bottom_right] = c3[bottom_right]

        blue[top_left] = c3[top_left]
        blue[top_right] = c1[top_right]
        blue[bottom_left] = c2[bottom_left]
        blue[bottom_right] = img[bottom_right]
    elif cfa == 'bggr':
        blue[top_left] = img[top_left]
        blue[top_right] = c2[top_right]
        blue[bottom_left] = c1[bottom_left]
        blue[bottom_right] = c3[bottom_right]

        red[top_left] = c3[top_left]
        red[top_right] = c1[top_right]
        red[bottom_left] = c2[bottom_left]
        red[bottom_right] = img[bottom_right]
    else:
        raise ErrBadCFA

    return np.stack((red, green, blue), axis=2)
