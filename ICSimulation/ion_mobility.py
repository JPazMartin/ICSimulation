"""
    Functions for calculating the positive and negative ions mobilities in air 
    from:
    Zhang et al. Prediction of Average Mobility of Ions from Corona Discharge in
    Air with Respect to Pressure, Humidity and Temperature" IEEE Transactions on
    Dielectrics and Electrical Insulation Vol. 26, No. 5; October 2019
    (10.1109/TDEI.2019.008001)
"""

from math import sqrt, exp

def RH_to_AH(T, RH):
    
    T += 273.15
    
    if T >= 0:
        
        AH = 10**((10.286 * T - 2148.4909) / (T - 35.85)) / 1000
        
    else :
        
        AH = 10**(12.5633 - 2670.59 / 273.1) / 1000
        
    return RH * AH / 100

def alpha_pos(T, H):
    
    T += 273.15
        
    A_1 = 1.438E-4 * T**(0.451)
    A_2 = 1.964E-4 * T**(0.513)
    A_3 = -0.95053 - exp((292.3191 - T) / 12.49447)
    H_0 = 0.05285 + exp((T - 301.5777) / 18.99722)
    
    return A_1 + (A_2 - A_1) / (1 + 10**(A_3 * (H_0 - H)))


def beta_pos(T, H):
    
    T += 273.15
        
    B_1 = -0.55579 - exp((T - 341.1570) / 21.32626)
    B_2 = 7.84584 - 0.06263 * T + 1.16655E-4 * T**2
    B_3 = 0.0377 + 0.144 / (1 + 10**((278 - T) / 16.8))
    
    
    return B_1 + B_2 * exp(-H / B_3)


def alpha_neg(T, H):
    
    T += 273.15
        
    A_1 = 2.476E-4 * T**(0.377)
    A_2 = 2.359E-4 * T**(0.499)
    A_3 = -0.79778 - exp((283.4930 - T) / 9.22739)
    H_0 = -0.04443 + exp((T - 301.1779) / 27.45497)
    
    return A_1 + (A_2 - A_1) / (1 + 10**(A_3 * (H_0 - H)))


def beta_neg(T, H):
    
    T += 273.15
        
    B_1 = -0.57882 - exp((T - 370.5096) / 46.72047)
    B_2 = 2.31325 - 0.02408 * T + 4.93039E-5 * T**2
    B_3 = 0.00152 + 0.112 / (1 + 10**((269 - T) / 21.3))
    
    return B_1 + B_2 * exp(-H / B_3)

def mob_pos(RH, P, T):
    
    H = RH_to_AH(T, RH)
        
    P = P / 10
    
    return 1 / sqrt(T + 273.15) * alpha_pos(T, H) * (3 * P / (T + 273.15))**beta_pos(T, H)
    
def mob_neg(RH, P, T):
        
    H = RH_to_AH(T, RH)
        
    P = P / 10
    
    return 1 / sqrt(T + 273.15) * alpha_neg(T, H) * (3 * P / (T + 273.15))**beta_neg(T, H)