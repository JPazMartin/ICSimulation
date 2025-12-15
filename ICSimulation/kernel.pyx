import cython
# cimport numpy as np
import  numpy as np

cdef double transport(const double[::1] charge, double[::1] chargeNew, const double mobility, const double[::1] eField,
              const int direction, const double[::1] area, const double rel) noexcept:

    """
    Function to transport the charge densities in a given direction.
    """

    cdef:
        int n = charge.shape[0]
        int i

    if direction == 1:

        for i in range(1, n):

            chargeNew[i] += (charge[i - 1] * mobility * eField[i - 1] * area[i - 1] / area[i] 
                                                    - charge[i] * mobility * eField[i]) * rel

        chargeNew[0] -= chargeNew[0] * mobility * eField[0] * rel

        return area[n - 1] * charge[n - 1] * mobility * eField[n - 1]

    if direction == -1:

        for i in range(n - 1):

            chargeNew[i] += (charge[i + 1] * mobility * eField[i + 1] * area[i + 1] / area[i] 
                                                - charge[i] * mobility * eField[i]) * rel

        chargeNew[n - 1] -= chargeNew[n - 1] * mobility * eField[n - 1] * rel

        return area[0] * charge[0] * mobility * eField[0]

cdef double linear_extrapolation(const double[::1] x_theo, const double[::1] y_theo, const double x_ex, const unsigned int n) noexcept:
    
    """
    Function to perform a linear interpolation.

    """
    
    cdef:
        
        unsigned int i;
        double   y = y_theo[0];
        
    if x_ex > x_theo[n - 1]:
        
        return x_ex * (y_theo[n - 1] - y_theo[n - 2]) / (x_theo[n - 1] - x_theo[n - 2]) + (x_theo[n - 1] * y_theo[n - 2] - x_theo[n - 2] * y_theo[n - 1]) / (x_theo[n - 1] - x_theo[n - 2])
    
    for i in range(1, n):
        
        if x_ex < x_theo[i]:
        
            if x_ex > x_theo[i - 1]:
            
                y = y_theo[i - 1] + (x_ex - x_theo[i - 1]) / (x_theo[i] - x_theo[i - 1]) * (y_theo[i] - y_theo[i - 1])
                
                return y
            
        if x_ex == x_theo[i]:
            
            return y_theo[i]
            
    return y

cpdef runSimulation(double[::1] eField, const double[::1] eField0, const double[::1] area,
                    const double[:, :] eParameters, const double stepE, const double alpha, 
                    const int eFieldP, const double dx, const int eSign, const double voltage,
                    const double k_pos, const double k_neg, const int eMult, const double cRelease,
                    const double n0, const double pulseDuration, const int timeStructFlag, 
                    const double[::1] timeStruct, const double[::1] n0Struct, double[:, :] current):

    cdef:

        int n        = eField.shape[0]
        int nCurr    = current.shape[0]
        int nStruct  = timeStruct.shape[0]
        int factor   = 1
        int idxArray = -1

        double e    = 1.6021766208E-19 # C
        double e0   = 8.854187817E-12  # F m^{-1}
        double er   = 1.000589

        double[::1] totalCharge = np.empty(n)
        double[::1] eVelocity   = np.empty(n)

        # Charge density arrays.
        double[::1] nE      = np.zeros(n)
        double[::1] nPos    = np.zeros(n)
        double[::1] nNeg    = np.zeros(n)
        double[::1] nENew   = np.zeros(n)
        double[::1] nNegNew = np.zeros(n)
        double[::1] nPosNew = np.zeros(n)

        double chargePos = 0
        double chargeNeg = 0
        double chargeE   = 0
        double nSum      = cRelease
        double eSum      = 0
        int idxMaxE      = 0
        double time      = 0
        double tStep     = min(1E-15, pulseDuration / 2000)
        double rel       = tStep / dx
        int index        = -1

        unsigned int i, idxBin
        double recomb, eFieldSum, irradiation;

    if pulseDuration == 0:

        for i in range(n):
            nENew[i]   = n0
            nPosNew[i] = n0

    while (nSum / cRelease > 1E-10) or (time < pulseDuration):

        time     += tStep
        rel       = tStep / dx
        index    += 1

        nSum = 0
        eSum = 0
        
        irradiation = 0
        if time <= pulseDuration and pulseDuration > 0:

            if timeStructFlag:
                irradiation = linear_extrapolation(timeStruct, n0Struct, time, nStruct) * tStep

            else:
                irradiation = n0 * tStep / pulseDuration

        if index % factor == 0:

            idxArray += 1

            if idxArray >= nCurr:

                factor   += 1
                idxArray  = int((nCurr - 1) / factor) + 1
                
                print(factor, current[:idxArray, :].shape, nCurr / factor, current[::factor, :].shape)

                current[:idxArray, :] = current[::factor, :]
                current[idxArray:, :] = 0
                
                idxArray += 1

        for i in range(n):

            idxBin = int(eField[i] / stepE + 0.5)

            ### -- Update arrays ---
            nPos[i] = nPosNew[i]
            nNeg[i] = nNegNew[i]
            nE[i]   = nENew[i]

            ### -- 1st step: Irradiation --
            nENew[i]   += irradiation
            nPosNew[i] += irradiation

            ### -- 2nd step: Ion-ion Recombination --
            recomb      = alpha * nNeg[i] * nPos[i] * tStep
            nNegNew[i] -= recomb
            nPosNew[i] -= recomb

            ## -- 3rd step: Attachment --
            attachment  = nE[i] * tStep / eParameters[idxBin, 2]
            nNegNew[i] += attachment
            nENew[i]   -= attachment

            # -- 4th step: Multiplication --
            if eMult == 1:

                multiplication = nE[i] * eParameters[idxBin, 3] * tStep
                
                nENew[i]   += multiplication
                nPosNew[i] += multiplication

            eSum += nENew[i] * area[i] * dx
            nSum += (nENew[i] + nPosNew[i] + nNegNew[i]) * area[i] * dx

            if index % factor == 0:

                current[idxArray, 0]  = time
                current[idxArray, 1] += e * eParameters[idxBin,1] * nE[i] * area[i] / n
                current[idxArray, 2] += e * k_pos * eField[i] * nPos[i] * area[i] / n
                current[idxArray, 3] += e * k_neg * eField[i] * nNeg[i] * area[i] / n

            eVelocity[i] = eParameters[idxBin,1]

        ### -- 5th step: transport of the charge --
        chargePos += transport(nPos, nPosNew, k_pos, eField, eSign, area, tStep / dx) * tStep
        chargeNeg += transport(nNeg, nNegNew, k_neg, eField, -eSign, area, tStep / dx) * tStep
        chargeE   += transport(nE, nENew, 1, eVelocity, -eSign, area, tStep / dx) * tStep

        ### -- 6th step: Electric field perturbation --
        if eFieldP == 1:

            eFieldSum = 0
            totalCharge[0] = (nPos[0] - nE[0] - nNeg[0]) * area[0] * dx

            for i in range(1, n):
            
                totalCharge[i] = totalCharge[i - 1] + (nPos[i] - nE[i] - nNeg[i]) * area[i] * dx
            
            for i in range(n):
                
                eField[i] = (e / (er * e0 * area[i])) * eSign * totalCharge[i]
                eFieldSum += eField[i]

            for i in range(n):
                
                eField[i] += (abs(voltage) - eFieldSum * dx) * eField0[i] / abs(voltage)

        ## If pulse is finished and electron charge is < 1E-10 the zero it 
        ## to be able to increase the time step.

        if eSum / cRelease < 1E-10 and time > pulseDuration: 
            for i in range(n): nENew[i] = 0

        ### -- Update time step --
        idxMaxE = 0
        for i in range(n):
            if eField[i] > eField[idxMaxE]: idxMaxE = i
            
        if eSum / cRelease > 1E-15 or (time < pulseDuration):
            tStep = 0.4 * dx / max(eVelocity[idxMaxE], eField[idxMaxE] * k_neg)
        else: tStep = 0.4 * dx / eField[idxMaxE] / k_neg
        
    return chargePos, chargeNeg, chargeE
