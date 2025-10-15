"""
Given a chief orbit and delta semi-major axis and delta mean anomaly at epoch,
compute the mean anomaly difference at a later time using the linearized
mean anomaly difference equation of motion 
"""

import numpy as np

from spacecraft_dynamics.orbit import Orbit
from spacecraft_dynamics.utils import true_to_mean_anomaly


def main():

    # Chief orbit params
    a = 7500 * 1000.0 # [m]
    e = 0.01
    
    # Assume true anomaly at epoch is 0
    f0 = 0
    M0 = true_to_mean_anomaly(f0, e)

    # Initial mean anomaly difference between chief and deputy
    delM0 = np.radians(15.0)

    # Semimajor axis difference (this is time invariant)
    delSemiMajorAxis = 50.0 * 1000 # [m]

    # Other params don't matter for this problem
    i = 0.0
    raan = 0.0
    argP = 0.0

    chiefOrbit = Orbit(a,e,i,raan,argP,M0,0)

    # Time to propagate the orbit to
    tFinal = 14400 # [s]

    # We don't need the orbit here, we're just computing the new delta mean anomaly
    # using the delta mean anomaly rate
    # dM = dM0 + dMdot * (tf - t0)
    delMdot = (-3. / 2.) * np.sqrt(chiefOrbit.mu / (a**3)) * delSemiMajorAxis / a
    delMFinal = delM0 + (delMdot * (tFinal - 0))
    print(f"Final delta mean anomaly: {np.degrees(delMFinal)} [deg]")

if __name__ == "__main__":
    main()