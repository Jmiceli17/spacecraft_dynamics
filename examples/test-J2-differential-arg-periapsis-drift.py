
import numpy as np

from spacecraft_dynamics.utils import constants


def main():

    # Define differential orbit elements and chief orbit elements
    sma = 8000.0 * 1000.0 # [m]
    i = np.radians(33.0) 
    e = 0.1
    delta_i = np.radians(0.1)


    # Non-dimensional semi-major axis term
    L = np.sqrt(sma / constants.R_EARTH)
    eta = np.sqrt(1.0 - (e * e))

    # Compute non-dimensional delta_omega_prime term
    delta_omega_prime = constants.J2_EARTH * (3.0 / (4.0 * L**7 * eta**4)) * (np.tan(i) * (5.0 * np.cos(i)**2 - 1.0) - 5.0 * np.sin(2.0 * i)) * delta_i

    # Convert to dimensional term
    n_eq = np.sqrt(constants.MU_EARTH_M / (constants.R_EARTH**3))
    delta_omega_dot = delta_omega_prime * n_eq

    print(f"|delta_omega_dot| = {np.abs(np.rad2deg(delta_omega_dot))} [deg/s]")


if __name__ == "__main__":
    main()