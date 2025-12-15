import numpy  as np
import kernel as k

from .ion_mobility       import mob_pos, mob_neg
from importlib.resources import files


## Some fundamental constants
e           = 1.6021766208E-19 # C
e0          = 8.854187817E-12  # F m^{-1}
er          = 1.000589
air_density = 1.204            # kg/m^3 at 20 degree celsius and 1013.25 hPa
Wair        = 33.97            # average energy per ion pair in eV.

## Reference temperature and pressure
refTemperature = 293.15  # K
refPressure    = 1013.25 # hPa

# Load electron properties
dataPath = files("ICSimulation.data").joinpath("dataElectrons.txt")
eTable   = np.loadtxt(dataPath)

def default_vele(E, humidity, pressure, temperature):

    """
    Default function for the electron velocity obtained using
    Magboltz simulation code.

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron velocity in m s^{-1}.
    """

    k_TP = (273.15 + temperature) / 293.15 * 1013.25 / pressure

    return np.interp(E, eTable[:, 0] / k_TP, eTable[:, 1])

def default_taue(E, humidity, pressure, temperature):

    """
    Default function for the electron attachment time
    from Boissonnat et al. (arXiv:1609.03740v1)

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron attachment time in s.
    """

    return 95.24E-9 * (1 - np.exp(-E / 258.5E3))

def default_mult(E, humidity, pressure, temperature):

    """
    Default function for the electron multiplication
    obtained using Magboltz code.

    *-- Inputs:

        :eField:      Electric field in V m^{-1}.
        :temperature: Temperature in degree Celsius.
        :pressure:    Pressure in hPa.
        :humidity:    Relative humidity in %.

    *-- Returns:

        Electron multiplication rate in s^{-1}.
    """

    k_TP = (273.15 + temperature) / 293.15 * 1013.25 / pressure

    return np.interp(E, eTable[:, 0] / k_TP, eTable[:, 2] / k_TP)

def CICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                        humidity, r1, r2, h, n, Ndw, eFieldP = 1, eMult = 1, mu_pos = mob_pos,
                        mu_neg = mob_neg, ve = default_vele, taue = default_taue,
                        multe = default_mult, timeStruct = [], doseRateStruct = []):

    """
    This function simulates the charge transport in a 1D cylindrical ionization
    chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :r1:            Internal radius of the ionization chamber in m.
        :r2:            External radius of the ionization chamber in m.
        :h:             Height of the ionization chamber in m.
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
        :multe:         Optional. Electron multiplication rate in s^{-1}. The
                        program requires a function of the electric field (in
                        V m^{-1}), relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = (r2 - r1) / n                                      # m
    x      = np.array([r1 + dx / 2 + i * dx for i in range(n)]) # m
    area   = 2 * np.pi * h * x                                  # m^2
    volume = np.pi * (r2**2 - r1**2) * h                        # m^3
    length = r2 - r1                                            # m    

    # Unperturbed electric field
    eField0 = abs(voltage) / (x * np.log(r2 / r1)) # V/m

    # Sign of the voltage
    eSign = np.sign(voltage)
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, eFieldP,
            eMult, mu_pos, mu_neg, ve, taue, multe, timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / refTemperature * refPressure / pressure

    # Charge released in the medium
    n0 = dpp / (e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * e * k_TP       # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def SICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                        humidity, r1, r2, n, Ndw, eFieldP = 1, eMult = 1, mu_pos = mob_pos,
                        mu_neg = mob_neg, ve = default_vele, taue = default_taue,
                        multe = default_mult, timeStruct = [], doseRateStruct = []):
    """
    This function simulates the charge transport in a 1D spherical ionization
    chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :r1:            Internal radius of the ionization chamber in m.
        :r2:            External radius of the ionization chamber in m.
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
        :multe:         Optional. Electron multiplication rate in s^{-1}. The
                        program requires a function of the electric field (in
                        V m^{-1}), relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = (r2 - r1) / n                                      # m
    x      = np.array([r1 + dx / 2 + i * dx for i in range(n)]) # m
    area   = 4 * np.pi * x**2                                   # m^2
    volume = 4 / 3 * np.pi * (r2**3 - r1**3)                    # m^3
    length = r2 - r1                                            # m    

    # Unperturbed electric field
    eField0 = - abs(voltage) / (x**2 * (1 / r2 - 1 / r1)) # V/m

    # Sign of the voltage
    eSign = np.sign(voltage)
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, eFieldP,
            eMult, mu_pos, mu_neg, ve, taue, multe, timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / refTemperature * refPressure / pressure

    # Charge released in the medium
    n0 = dpp / (e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * e * k_TP       # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def PPICpulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                         humidity, d, radius, n, Ndw, eFieldP = 1, eMult = 1, mu_pos = mob_pos,
                         mu_neg = mob_neg, ve = default_vele, taue = default_taue,
                         multe = default_mult, timeStruct = [], doseRateStruct = []):
    
    """
    This function simulates the charge transport in a 1D parallel plate ionization
    chamber.

    *-- Inputs:
        
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
        :multe:         Optional. Electron multiplication rate in s^{-1}. The
                        program requires a function of the electric field (in
                        V m^{-1}), relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :CCE:    Charge collection efficiency.
        :FEF0:   Free electron fraction in relation to the released charge in the medium.
        :FEF1:   Free electron fraction in relation to the collected charge.
        :Q_col:  Collected charge per pulse referenced to 20 ºC and 1013.25 hPa.
        :I:      Array with the time in s and intensity in A for each of the
                 three species considered.
                    . I[:, 0] -> Array with the times in s.
                    . I[:, 1] -> Array with the instantaneous current from
                                 electrons in A.
                    . I[:, 2] -> Array with the instantaneous current from
                                 positive ions in A.
                    . I[:, 3] -> Array with the instantaneous current from
                                 negative ions in A.
    """

    # Chamber dimension properties.
    dx     = d / n                                         # m
    x      = np.array([dx / 2 + i * dx for i in range(n)]) # m
    area   = np.pi * radius**2 * np.ones(n)                 # m^2
    volume = np.pi * radius**2 * d                          # m^3
    length = d                                             # m    

    # Unperturbed electric field
    eField0 = abs(voltage) * np.ones(n) / d # V/m

    # Sign of the voltage
    eSign = 1
 
    pars = [dpp, pulseDuration, alpha, voltage, temperature, pressure, humidity,
            area, volume, length, eField0, dx, x, eSign, n, Ndw, eFieldP, 
            eMult, mu_pos, mu_neg, ve, taue, multe, timeStruct, doseRateStruct]
    
    chargePos, chargeNeg, chargeE, I = pulsedSimulation(*pars)

    k_TP = (273.15 + temperature) / refTemperature * refPressure / pressure
    
    # Charge released in the medium
    n0 = dpp / (e * Ndw * volume * k_TP) # m^-3

    CCE   = chargePos / (n0 * volume)  # Charge collection efficiency
    FEF0  = chargeE   / (n0 * volume)  # FEF (Boag definition 0)
    FEF1  = chargeE   / chargePos      # FEF (Boag definition 1)
    Qcoll = chargePos * e * k_TP     # Charge referenced to STP

    return CCE, FEF0, FEF1, Qcoll, I

def pulsedSimulation(dpp, pulseDuration, alpha, voltage, temperature, pressure,
                     humidity, area, volume, length, eField0, dx, x, eSign, n,
                     Ndw, eFieldP = 1, eMult = 1, mu_pos = mob_pos,
                     mu_neg = mob_neg, ve = default_vele, taue = default_taue,
                     multe = default_mult, timeStruct = [], doseRateStruct = []):
    
    """
    This function generically simulates the charge transport in a 1D ionization
    chamber. The parameters area and eField0 will essentially determine the
    symmetry of the ionziation chamber.

    *-- Inputs:
        
        :dpp:           Dose per pulse in Gy.
        :pulseDuration: Pulse duration in s.
        :alpha:         Volume recombination constant in m^{3} s^{-1}.
        :voltage:       Voltage in V.
        :temperature:   Temperature in degree Celsius.
        :pressure:      Pressure in hPa.
        :humidity:      Relative humidity in %.
        :area:          Area of the elements m^2. This parameter must be an
                        array.
        :volume:        Ionization chamber volume in m^-3.
        :length:        Length of the geometry in m.
        :eField0:       Initial electric field in V/m. This parameters must be 
                        an array.
        :dx:            Distance between the elements.
        :x:             Coordinates.
        :eSign:         Direction of the charge transport.
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
        :multe:         Optional. Electron multiplication rate in s^{-1}. The
                        program requires a function of the electric field (in
                        V m^{-1}), relative humidity (in %), pressure (in hPa)
                        and temperature (in degree Celsius), following this order.
        :timeStruct:    Optional, array with the time in s for a custom pulse
                        structure.
        :DoseStruct:    Optional, array with the dose rate in arbitrary units
                        for a custom pulse structure.

    *-- Returns:

        :chargePos:  "Collected" charge from positive ions in C.
        :chargeNeg:  "Collected" charge from negative ions in C.
        :chargeE:    "Collected" charge from electron in C.
        :I:          Array with the time in s and intensity in A for each of the
                     three species considered.
                         . I[:, 0] -> Array with the times in s.
                         . I[:, 1] -> Array with the instantaneous current from
                                      electrons in A.
                         . I[:, 2] -> Array with the instantaneous current from
                                      positive ions in A.
                         . I[:, 3] -> Array with the instantaneous current from
                                      negative ions in A.
    """

    # Negative and positive ion mobilities (now supplied by the user as a function of h, p, t)
    k_neg =  mu_neg(humidity, pressure, temperature)
    k_pos =  mu_pos(humidity, pressure, temperature)

    k_TP  = (273.15 + temperature) / refTemperature * refPressure / pressure

    # Charge released in the medium
    timeStructFlag = 0
    if len(timeStruct) > 0: 
        timeStructFlag  = 1
        doseRateStruct *= dpp / np.trapz(doseRateStruct, timeStruct)
        n0Struct        = doseRateStruct / (e * Ndw * volume * k_TP) # m^-3
        pulseDuration   = timeStruct[-1]
        n0              = np.trapz(n0Struct, timeStruct)
    else:
        n0 = dpp / (e * Ndw * volume * k_TP) # m^-3
        n0Struct   = np.array([])
        timeStruct = np.array([])

    # Perturbed electric field
    eField  = np.copy(eField0) # V/m

    ## Arrays for electron parameters
    nTables     = 100000
    maxE        = 4.9E6 # V m^{-1}
    stepE       = maxE / nTables
    eParameters = np.zeros([nTables, 4])

    eParameters[:, 0] = stepE * np.arange(nTables)
    eParameters[:, 1] = ve(eParameters[:, 0], humidity, pressure, temperature)
    eParameters[:, 2] = taue(eParameters[:, 0], humidity, pressure, temperature)
    eParameters[:, 3] = multe(eParameters[:, 0], humidity, pressure, temperature) * eParameters[:, 1]

    # "Collected" charges
    chargeE = 0.0; chargePos = 0.0; chargeNeg = 0.0
    # Instantaneous induced current
    nMax    = 1000000
    current = np.zeros([nMax, 4])
    
    # Loop until no charge (nSum > 1) is left in the simulation.
    chargePos, chargeNeg, chargeE = k.runSimulation(eField, eField0, area, eParameters,
                                     stepE, alpha, eFieldP, dx, eSign, voltage, k_pos,
                                     k_neg, eMult, 2 * n0 * volume, n0, pulseDuration,
                                     timeStructFlag, timeStruct, n0Struct, current)
    
    current = current[current[:, 0] != 0]

    return chargePos, chargeNeg, chargeE, current
 
