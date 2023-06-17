"""radiometric calculations"""
from prysm.mathops import np

C = 299_792_458 # m/s
BOLTZMANN_CONSTANT = 1.380_649*1e-23 # J / K
PLANCK_CONSTANT = 6.62_607_015*1e-34 # J * s

def radiance_from_blackbody_temp(T, wavelength):
    """compute the spectral radiance of a blackbody at a given wavelength and temperature

    Parameters
    ----------
    T : float
        blackbody temperature
    wavelength : float
        wavelength of light in microns

    Returns
    -------
    float
        spectral radiance of a blackbody in W/(Sr * m^2 * um)
        TODO : add option to integrate over wavelength to return radiance instead 
    """
    wavelength *= 1e-6 # convert to meters
    top = 2 * PLANCK_CONSTANT * C**2 / wavelength**5
    bottom = np.exp(PLANCK_CONSTANT * C / (wavelength * BOLTZMANN_CONSTANT * T)) - 1 
    out = top / bottom * 1e-6 
    return out

