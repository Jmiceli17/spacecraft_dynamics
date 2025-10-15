"""
Given a differential arg of periapsis that we want to correct with the implied change of delta_M = -delta_omega
find the delta V required to provide the correction
"""

import numpy as np
from spacecraft_dynamics.utils import constants
def main():

    deltaOmega = np.radians(0.1)
    sma = 8000.0 * 1000.0 # [m]
    e = 0.1
    eta = np.sqrt(1.0 - (e * e))
    meanMotion = np.sqrt(constants.MU_EARTH_M / (sma**3))

    # Assume impulses applied at periapsis and apoapsis and only in the radial direction so 
    # other equations of motion are not affected
    deltaVperiapsis = meanMotion * sma / 4.0 * (1.0 - ((1.0 + e)**2 / eta)) * deltaOmega
    deltaVapoapsis = meanMotion * sma / 4.0 * (1.0 - ((1.0 - e)**2 / eta)) * deltaOmega

    print(f"deltaVperiapsis: {deltaVperiapsis} [m/s]")
    print(f"deltaVapoapsis: {deltaVapoapsis} [m/s]")


if __name__ == "__main__":
    main()