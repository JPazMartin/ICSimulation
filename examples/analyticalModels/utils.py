"""
Useful functions to evaluate analytical models.

"""

import numpy as np

## *-- Analytical formulas needed:
def gpp(d):

    """
    Geometrical parameter for parallel-plate ionization
    chambers.

    *-- Inputs:

        :d: Distance between electrodes in m.

    *-- Returns:

        Value of the geometrical parameter in m^2.
    """

    return d**2

def gcyl(r1, r2):

    """
    Geometrical parameter for cylindrical ionization
    chambers.

    *-- Inputs:

        :r1: Internal radius of the ionization chamber in m.
        :r2: External radius of the ionization chamber in m.
        
    *-- Returns:

        Value of the geometrical parameter in m^2.
    """

    return (r2**2 - r1**2) / 2 * np.log(r2 / r1)

def u(alpha, N0, g, k_pos, k_neg, voltage):

    """
    Boag's dimensionless combination of parameters from
    Equation [2] in Brit. J. Radiol. 23 (1950) 601-611.

    *-- Inputs:

        :alpha:   Volume recombination constant in m^{3}.
        :N0:      Carrier density in m^{-3}.
        :g:       Geometrical parameter.
        :k_pos:   Positive ion mobility in m^2 V^{-1} s^{-1}.
        :k_neg:   Negative ion mobility in m^2 V^{-1} s^{-1}.
        :voltage: Voltage in V.
        
    *-- Returns:

        Value of the dimensionless parameter.
    """

    return alpha * N0 * g / ((k_pos + k_neg) * voltage)

def N0(dpp):

    """
    Carrier density for a given dose per pulse, assuming 20 degree
    Celsius and 1013.25 hPa. It gives an approximate value for 
    the carrier density in air for a given dose per pulse
    assuming the perturbation factor and the stopping power ratio
    air / medium to be 1.

    *-- Inputs:

        :dpp:     Dose per pulse in Gy.
        
    *-- Returns:

        Charge carrier densities in m^{-3}.
    """

    return dpp * 1.204 / (1.6021766208E-19 * 33.97)

def boag0(u):

    """
    Boag's charge collection efficiency formula Equation [1]
    in Brit. J. Radiol. 23 (1950) 601-611.

    *-- Inputs:

        :u:       Boag's dimensionless parameter.
        
    *-- Returns:

        Charge collection efficeincy according to Boag's model.
    """

    return np.log(1 + u) / u


## *-- Transport parameters:
def k_pos(humidity, pressure, temperature):

    """
    Positive ion mobility from Boissonnat et al.
    (arXiv:1609.03740v1)

    *-- Inputs:

        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Positive ion mobility in m^2 V^{-1} s^{-1}.
    """

    return 1.87E-4 * (temperature + 273.15) / (293.15) * (1013.25 / pressure)

def k_neg(humidity, pressure, temperature):

    """
    Negative ion mobility from Boissonnat et al.
    (arXiv:1609.03740v1)

    *-- Inputs:

        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Negative ion mobility in m^2 V^{-1} s^{-1}.
    """

    return 2.09E-4 * (temperature + 273.15) / (293.15) * (1013.25 / pressure)