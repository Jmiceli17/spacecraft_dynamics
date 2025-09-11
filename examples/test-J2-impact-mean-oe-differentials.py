

import numpy as np
from spacecraft_dynamics.utils import constants
def main():

    # Chief orbit params
    sma = 8000.0 * 1.0e3 # km
    ecc = 0.1
    inc = np.radians(80.0)
    meanMotion = np.sqrt(constants.MU_EARTH_M / (sma * sma * sma))
    # OE differences of deputy
    da = 10.0 * 1.0e3
    de = 0.05
    di = np.radians(1.0)

    # Dimensionless J2 term
    J2 = 1.086262668e-03

    # Compute rate of change of mean ascending node difference using the first variation of dRaan / dt
    eta = np.sqrt(1.0 - (ecc * ecc))
    epsilon =  3.0 * J2 * (constants.R_EARTH / (sma * (1.0 - ecc * ecc)))**2
    dKappaRaan = (7.0 / 4.0 * np.cos(inc) * da / sma) - (2.0 * ecc / (eta * eta) * np.cos(inc) * de) + (1.0 / 2.0 * np.sin(inc) * di)

    # dRaanDot(t)
    dRaanRate = epsilon * meanMotion * dKappaRaan

    # Since the rate is constant we can determine the total change by multiplying it by the orbit period length
    # Orbit period [s]
    period = 2.0 * np.pi / meanMotion

    # Total change in RAAN difference [rad]
    dRaan = dRaanRate * period

    print(f"total change: {np.degrees(dRaan)} [deg]")

if __name__ == "__main__":
    main()