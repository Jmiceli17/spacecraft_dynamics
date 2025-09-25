import numpy as np
import math

from spacecraft_dynamics.utils import constants

def main():

    # Chief orbit elements
    a = 8000 * 1000 # [m]
    e = 0.1
    i = np.radians(33.0) # [rad]
    n = np.sqrt(constants.MU_EARTH_M / (a * a * a))
    p = a * (1.0 - e * e)
    # Dimensionless J2 term
    J2 = 1.086262668e-03

    # Calculate differential mean ascending node rate [rad/s]
    dRaanDt = -3 / 2 * J2 * n * (((constants.R_EARTH) / p)**2) * np.cos(i)

    # Time to complete one revolution
    revTime = 2 * math.pi / dRaanDt

    # Convert to days
    secPerDay = 86400
    revTime = revTime / secPerDay

    # Note negative value because rate is negative (precession)
    print(f"rev time: {abs(revTime)} # [days]")


if __name__ == "__main__":
    main()

