"""
    Comparison between the numerical code and the analytical model developed
    by Fenwick et al. in Phys. Med. Biol. 69 (2024) 155023 and Phys. Med. Biol.
    (2023) 015016.

"""

import numpy            as np
import matplotlib.pylab as plt
import os

from scipy.special       import exp1
from ICSimulation        import PPICpulsedSimulation, CICpulsedSimulation
from importlib.resources import files

## *-- Analytical formulas needed:

def gpp(d):

    """
    Geometrical parameter for parallel-plate ionization
    chambers from Phys. Med. Biol. (2023) 015016.

    *-- Inputs:

        :d: Distance between electrodes in m.

    *-- Returns:

        Value of the geometrical parameter in m^2.
    """

    return d**2

def gcyl(r1, r2):

    """
    Geometrical parameter for cylindrical ionization
    chambers from Phys. Med. Biol. (2023) 015016.

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

def h(u, D):

    """
    Equation [3] in Phys. Med. Biol. 69 (2024) 155023
    
    *-- Inputs:

        :u:   Boag's dimensionless parameter.
        :D:   Dimensionless delta parameter.
        
    *-- Returns:

        Value of the dimensionless parameter.
    """
    return 1 / D * np.exp(u / D) * (exp1(u * np.exp(-D) / D) - exp1(u / D))

def Fenwick(u, k_e, voltage, tau, g):
    
    """
    Charge collection efficiency from Fenwick et al. model.
    Corresponds to Equation [3] in Phys. Med. Biol. 69 (2024)
    155023.

    *-- Inputs:

        :u:       Boag's dimensionless parameter.
        :k_e:     Electron mobility.
        :voltage: Voltage in V.
        :tau:     Electron attachment time in s.
        :g:       Geometrical parameter
        
    *-- Returns:

        Charge collection efficiency using the Fenwick et al. model.
    """

    return 1 / u * np.log(1 + u * h(u, g / (k_e * voltage * tau)))

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

## *-- Transport parameters:
##     1.- Transport parameters from Table 1 in Phys. Med. Biol.
##         69 (2024) 155023: 

def ve_Table1(eField, humidity, pressure, temperature):

    """
    Electron velocity from Table 1 in Phys. Med. Biol. 69 (2024) 155023

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron velocity in m s^{-1}.
    """

    return 8.30E-2 * (temperature + 273.15) / (293.15) * (1013.25 / pressure) * eField

def k_pos_Table1(humidity, pressure, temperature):

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

def k_neg_Table1(humidity, pressure, temperature):

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

def tau_Table1(eField, humidity, pressure, temperature):

    """
    Electron attachment rate from Table 1 in Phys.
    Med. Biol. 69 (2024) 155023

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron attachment time in s.

    """

    return np.where(eField >= 0.327E5, 1E-7 / (1.1 + 11.3 * np.exp(-1.04E-5 * eField)),
                    1 / (7.0E7 + 657 * eField))


# Volume recombination parameter from Table 1 in Phys. Med. Biol. 69 (2024) 155023
alpha  = 1.30E-12 # m^3 s^{-1}

##     2.- Electron transport parameters obtained 
##         using Magboltz.

## Load the transport parameter table
dataPath = files("ICSimulation.data").joinpath("dataElectrons.txt")
eTable   = np.loadtxt(dataPath)

def ve_Magboltz(E, h, p, t):

    """
    Electron velocity obtained using Magboltz
    simulation code.

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron velocity in m s^{-1}.
    """

    k_TP = (273.15 + t) / 293.15 * 1013.25 / p

    return np.interp(E, eTable[:, 0] / k_TP, eTable[:, 1])

##     3.- Other transport parameters:
def tau_Boissonnat(E, humidity, pressure, temperature):

    """
    Electron attachment time from Boissonnat et al.
    (arXiv:1609.03740v1)

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron attachment time in s.
    """

    return 95.24E-9 * (1 - np.exp(-E / 258.5E3))


## *-- Input parameters:
voltage = 300.0  # Bias voltage applied to the chamber in V.

r1      = 0.3E-3 # Internal radius in m.
r2      = 1.0E-3 # External radius in m.

# I use the same distance between electrodes to get the same
# geometrical parameter as with cylindrical IC.
d       = gcyl(r1, r2)**0.5

## Constant electron attachment time for cylindrical simulations !
def tau_constant(eField, humidity, pressure, temperature):

    """
    Constant electron attachment time with respect
    to the electric field to use in the simulations.

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron attachment time in s.
    """

    return tau_Table1(voltage / d, humidity, pressure, temperature)


## *-- Numerical simulations:
dpp_sim = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9] # in Gy
nSteps  = 2000

# Charge collection efficiencies simulated using parallel-plate
# ionization chamber and cilindrical ionization chamber.
CCE_simPP  = []; CCE_simCyl = []

for dpp in dpp_sim:

    print(f"Simulating {dpp:.1f} Gy")

    """
    Parallel-plate chamber simulation:

    *-- Input parameters are:

        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :d:             Distance between electrodes in m.
        :radius:        Radius of the sensitive volume of the ionization chamber
                        in m.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :eMult:         Optional. Set to 0 to disable electron multiplication
        :mu_pos:        Optional. Modify the default positive ion mobility in
                        m^2 V^{-1} s^{-1}. The simulation program requires a
                        function of relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :mu_neg:        Optional. Modify the default positive ion mobility in
                        m^2 V^{-1} s^{-1}. The simulation program requires a
                        function of relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :ve:            Optional. Electron velocity in m s^{-1}. The program
                        requires a function of the electric field (in V m^{-1}),
                        relative humidity (in %), pressure (in hPa) and 
                        temperature (in degree Celsius), following this order.
        :taue:          Optional. Electron attachment time in s. The program
                        requires a function of the electric field (in V m^{-1}),
                        relative humidity (in %), pressure (in hPa) and 
                        temperature (in degree Celsius), following this order.
    """
    inputParameters_pp = [dpp, 0.0, alpha, voltage, 20.0, 1013.25, 50.0, d,
                          (d * np.pi)**-0.5, nSteps, 33.97 / 1.204, 0, 0,
                          k_pos_Table1, k_neg_Table1, ve_Table1, tau_Table1]
    
    CCE, FEF0, FEF1, Q_coll, I = PPICpulsedSimulation(*inputParameters_pp)

    CCE_simPP.append(CCE)

    """
    Cylindrical chamber simulation:

    *-- Input parameters are:

        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :r1:            Internal radius of the ionization chamber in m.
        :r2:            External radius of the ionization chamber in m.
        :h:             Heigh of the ionization chamber in m.
        :n:             Number of elements in which the geometry is divided.
        :Ndw:           Calibration coefficient in Gy C^{-1}. The calibration
                        coefficient must include all the correction factors
                        that affect the released charge excluding k_TP.
        :eFieldP:       Optional. Set to 0 to disable electric field
                        perturbation.
        :eMult:         Optional. Set to 0 to disable electron multiplication
        :mu_pos:        Optional. Modify the default positive ion mobility in
                        m^2 V^{-1} s^{-1}. The simulation program requires a
                        function of relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :mu_neg:        Optional. Modify the default positive ion mobility in
                        m^2 V^{-1} s^{-1}. The simulation program requires a
                        function of relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :ve:            Optional. Electron velocity in m s^{-1}. The program
                        requires a function of the electric field (in V m^{-1}),
                        relative humidity (in %), pressure (in hPa) and 
                        temperature (in degree Celsius), following this order.
        :taue:          Optional. Electron attachment time in s. The program
                        requires a function of the electric field (in V m^{-1}),
                        relative humidity (in %), pressure (in hPa) and 
                        temperature (in degree Celsius), following this order.
    """

    inputParameters_cyl = [dpp, 0.0, alpha, voltage, 20.0, 1013.25, 50.0, r1, r2,
                           (np.pi * (r2**2 - r1**2))**-1, nSteps, 33.97 / 1.204, 0,
                           0, k_pos_Table1, k_neg_Table1, ve_Table1, tau_constant]
    
    CCE, FEF0, FEF1, Q_coll, I = CICpulsedSimulation(*inputParameters_cyl)

    CCE_simCyl.append(CCE)

CCE_simPP  = np.array(CCE_simPP)
CCE_simCyl = np.array(CCE_simCyl)

## *-- Parameters to evaluate the analytical formulas.
k_pos = k_pos_Table1(50.0, 1013.25, 20.0)
k_neg = k_neg_Table1(50.0, 1013.25, 20.0)

# Electron mobility in m^2 V^{-1} s^{-1}
k_e   = ve_Table1(1, 50.0, 1013.25, 20.0) 

# Electron attachment time in s^-1
tau   = tau_Table1(voltage / d, 50.0, 1013.25, 20.0)

## *-- Evaluate the analytical formulas:
dpp = np.linspace(0.1, 10, 100)

CCE_pp  = Fenwick(u(alpha, N0(dpp), gpp(d), k_pos, k_neg, voltage), k_e, 
                  voltage, tau, gpp(d))

CCE_cyl = Fenwick(u(alpha, N0(dpp), gcyl(r1, r2), k_pos, k_neg, voltage), k_e, 
                  voltage, tau, gcyl(r1, r2))

## *-- Plot the results:
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(dpp,  CCE_pp, "-k", label = "Analytical PPIC")
ax.plot(dpp, CCE_cyl, "--r", label = "Analytical CIC")

ax.plot(dpp_sim,  CCE_simPP, "s", color = "k", markeredgecolor = "k",
        markerfacecolor = "None", markersize = 6, markeredgewidth = 0.8, label = "Simplified simulation PPIC")

ax.plot(dpp_sim, CCE_simCyl, "o", color = "k", markeredgecolor = "k",
        markerfacecolor = "None", markersize = 6, markeredgewidth = 0.8, label = "Simplified simulation CIC")

ax.set_xlabel(r"Dose per pulse (Gy)")
ax.set_ylabel(r"Charge collection efficiency")

ax.minorticks_on()
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.8,  1.0])

leg = ax.legend(borderpad = 0.2, fontsize = 15)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square')

fig.tight_layout()
plt.show()

fig.savefig("Figure_exampleAnalyticalModel.pdf")

## *-- Report the maximum differences:
##     If the range of dose per pulse is change or the simulation
##     points are modify the index must be change accordinly.

idx = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]

print(f"Maximum difference (PPIC) = {np.max((CCE_simPP - CCE_pp[idx]) / CCE_simPP) * 100:.2f} %")
print(f"Maximum difference (CIC)  = {np.max((CCE_simCyl - CCE_cyl[idx]) / CCE_simCyl) * 100:.2f} %")

## With this parameters and number of steps the differences are below 0.03 %.
