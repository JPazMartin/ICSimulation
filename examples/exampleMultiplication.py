"""
    Example showing the behaviour of the electron multiplication

"""

import numpy            as np
import matplotlib.pylab as plt
import os

from ICSimulation  import PPICpulsedSimulation


## *-- Numerical simulations:
voltages      = np.linspace(10, 500, 20, endpoint = True)
nSteps        = 1000
d             = 0.25E-3 # m

# Simulated charge collection efficiencies
CCE_sim  = []

for voltage in voltages:

    print(f"Simulating ... {voltage:.0f} V")
    
    ## Parallel-plate simulation:
    inputParameters_pp = [1.0E-3, 0.0, 1.3E-12, voltage, 20.0, 1013.25, 50.0, d,
                          (d * np.pi)**-0.5, nSteps, 33.97 / 1.20]
    
    CCE, FEF0, FEF1, Q_coll, I = PPICpulsedSimulation(*inputParameters_pp, eMult = 1)

    CCE_sim.append(CCE)

CCE_sim  = np.array(CCE_sim)

## *-- Plot the results:
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(voltages, CCE_sim, "-s", color = "k", markeredgecolor = "k",
        markerfacecolor = "None", markersize = 6, markeredgewidth = 0.8)

ax.set_xlabel(r"Applied bias voltage (V)")
ax.set_ylabel(r"Charge collection efficiency")

ax.minorticks_on()
ax.set_xlim([0.0, 500])
ax.set_ylim([0.99, 1.025])

fig.tight_layout()
plt.show()

fig.savefig("Figure_exampleMultiplication.pdf")
