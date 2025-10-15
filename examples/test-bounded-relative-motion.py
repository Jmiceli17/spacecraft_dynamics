import numpy as np

from spacecraft_dynamics.utils import constants

def main():

    # Assuming chief orbiting Earth in elliptic orbit, given x component of position
    # at periapsis, determine the required y component of velocity to maintain
    # bounded relative motion
    
    e = 0.1
    a = 8000 # [km]
    x0 = -10 # [km]
    
    n = np.sqrt(constants.MU_EARTH_KM / (a * a * a))

    y0dot = (-n * (2 + e) ) / np.sqrt((1 + e) * ((1 - e)**3)) * x0

    print(f"Required y0dot: {y0dot}")


if __name__ == "__main__":
    main()