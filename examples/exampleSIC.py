"""
    Example of the usage of the module for a spherical ionization chamber.

    Explanation of the variables in detail:

    dpp           : Dose per pulse in Gy.
    alpha         : Volume recombination parameter in m^{3} s^{-1}. Usually we 
                    are using a value between 0.9E-12 m^{3} s^{-1} up to 1.5E-12 m^{3} s^{-1}.
    pulseDuration : Pulse duration of the beam in s.
    temperature   : Temperature of the air in the ionization chamber in degree Celsius.
    pressure      : Pressure of the air inside the ionization chamber in hPa.
    rHumidity     : Relative humidity of the air in the ionization chamber in %.
    voltage       : Bias applied voltage in V. For spherical and cylindrical IC
                    the CCE may depend on the sign of this value.
    r1            : Internal radius of the cylindrical ionization chamber in m.
    r2            : External radius of the cylindrical ionization chamber in m.
    Ndw           : Calibration coefficient of the ionization chamber in Gy C^{-1}.
                    The calibration coefficient must have applied all the factor
                    related to the charge released in the medium but not the
                    temperature and pressure correction which the simulation will
                    take into account.
    n             : Number of discretization steps in position. A reasonable number
                    is around 1000. You may increase to have lower numerical error.
    fig           : If true, a figure will be display with the electric field and the
                    charge densities
    eFieldP       : Flag to activate/deactivate the electric field perturbation.
                    By default it is activated.

    +- Optional arguments:
    eFieldP       : Flag to activate/deactivate the electric field perturbation.
                    By default it is activated.
    tStruct       : Array with the time in s for a custom pulse structure.
    dStruct       : Array with the dose rate in arbitrary units for a custom pulse
                    structure. The values will be normalized internally to fulfill
                    int(dStruct * dt) = dpp 
"""

import time
import matplotlib.pylab as plt

from ICSimulation import SICpulsedSimulation

dpp           = 1        # Gy
alpha         = 1.3E-12  # m^3/s
pulseDuration = 1.0E-6   # s
temperature   = 20.0     # degree celsius
pressure      = 1013.25  # hPa
rHumidity     = 50       # %
voltage       = -200     # V
r1            = 0.500E-3 # m
r2            = 1.000E-3 # m
n             = 1000
eFieldP       = 1        # Electric field perturbation activated
Ndw           = 7.7E9    # Gy/C

inputParameters = [dpp, pulseDuration, alpha, voltage, temperature,
                    pressure, rHumidity, r1, r2, n, Ndw, eFieldP]

t0 = time.time()
CCE, FEF0, FEF1, Q_coll, I = SICpulsedSimulation(*inputParameters)

# This function returns:
#
# CCE    : Charge collection efficiency
# FEF0   : Free electron fraction in relation to the released charge in the 
#          medium.
# FEF1   : Free electron fraction in relation to the collected charge.
# Q_coll : Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
# I      : Array with the times and the  induced current by each charge carrier
#          in A

print(f"CCE    = {CCE:.4f}")
print(f"FEF0   = {FEF0:.4f}")
print(f"FEF1   = {FEF1:.4f}")
print(f"Q_coll = {Q_coll * 1E9:.4f} nC\n")

print(f"Elapsed time = {time.time() - t0:.4f} s")

# Plot the instantaneous induced current:
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(I[:, 0] * 1E6, I[:, 1] * 1E3, "-r", label = "Electrons")
ax.plot(I[:, 0] * 1E6, I[:, 2] * 1E3, "-b", label = "Positive ions")
ax.plot(I[:, 0] * 1E6, I[:, 3] * 1E3, "-m", label = "Negative ions")

ax.set_xlabel(r"Time (us)")
ax.set_ylabel(r"Intensity (mA)")

ax.set_ylim([1E-4, 1E0])
ax.set_xlim([1E-6, 1E3])

ax.set_xscale("log")
ax.set_yscale("log")

fig.tight_layout()
plt.show()

fig.savefig("Figure_exampleSIC.pdf")
