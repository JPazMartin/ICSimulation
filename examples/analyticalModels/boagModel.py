"""
    Comparison between the numerical code and the analytical model developed
    by Boag in Boag JW. Br J Radiol. 1950;23(274):601-611 for paralle-plate
    ionization chambers.

"""

import matplotlib.pylab as plt
import numpy            as np
import utils            as u

from ICSimulation import PPICpulsedSimulation


## *-- Numerical simulations:
dpp_sim = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9] # in Gy
nSteps  = 2000
alpha   = 1.30E-12 # m^3 s^{-1}
voltage = 300      # V
d       = 1.00E-3  # m

# Charge collection efficiencies simulated using a parallel-plate ionization chamber.
CCE_simPP  = []

for dpp in dpp_sim:

    print(f"Simulating {dpp:.1f} Gy")

    # Parallel-plate chamber simulation:

    inputParameters_pp = [dpp, 0.0, alpha, voltage, 20.0, 1013.25, 50.0, d,
                          (d * np.pi)**(-0.5), nSteps, 33.97 / 1.204, alpha, 0, 0,
                          u.k_pos, u.k_neg, lambda E, h, p, t: u.k_neg(h, p, t) * E,
                          lambda E, h, p, t: 1E10]
    
    CCE, FEF0, FEF1, Q_coll, I = PPICpulsedSimulation(*inputParameters_pp)

    CCE_simPP.append(CCE)

CCE_simPP  = np.array(CCE_simPP)

## *-- Parameters to evaluate the analytical formulas.
k_pos = u.k_pos(50.0, 1013.25, 20.0)
k_neg = u.k_neg(50.0, 1013.25, 20.0)

## *-- Evaluate the analytical formulas:
dpp = np.linspace(0.1, 10, 100)

CCE_pp  = u.boag0(u.u(alpha, u.N0(dpp), u.gpp(d), k_pos, k_neg, voltage))

## *-- Plot the results:
fig, ax = plt.subplots(figsize = (6, 6))

ax.plot(dpp,  CCE_pp, "-k", label = "Analytical PPIC")

ax.plot(dpp_sim,  CCE_simPP, "s", color = "k", markeredgecolor = "k",
        markerfacecolor = "None", markersize = 6, markeredgewidth = 0.8, label = "Simplified simulation PPIC")

ax.set_xlabel(r"Dose per pulse (Gy)", fontsize = 15)
ax.set_ylabel(r"Charge collection efficiency", fontsize = 15)

ax.minorticks_on()
ax.set_xlim([0.0, 10.0])
ax.set_ylim([0.0,  1.0])

leg = ax.legend(borderpad = 0.2, fontsize = 15)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square')

fig.tight_layout()
plt.show()

fig.savefig("Figure_exampleBoagModel.pdf")

## *-- Report the maximum differences:
##     If the range of dose per pulse is change or the simulation
##     points are modify the index must be change accordinly.

idx = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89]

print(f"Maximum difference (PPIC) = {np.max((CCE_simPP - CCE_pp[idx]) / CCE_simPP) * 100:.2f} %")

## With this parameters and number of steps the differences are below 0.03 %.
