"""
Comparing cartesian center of mass with orbit element center of mass
"""

import numpy as np
from spacecraft_dynamics.orbit import Orbit
from spacecraft_dynamics.utils import true_to_mean_anomaly

def main():
    # Initial orbit elements
    a = 7000*1000.0 # m
    e = 0.01
    i = np.radians(78.0)
    raan = np.radians(120.0)
    argP = np.radians(33.0)
    trueAnom = np.radians(45.0)

    # Deputy is in leader-follower formation
    delTrueAnom = np.radians(1.0)

    # Convert true anomaly to mean anomaly
    meanAnomAtEpochChief = true_to_mean_anomaly(trueAnom, e)
    meanAnomAtEpochDeputy = true_to_mean_anomaly(trueAnom + delTrueAnom, e)

    orbitChief = Orbit(a,e,i,raan,argP,meanAnomAtEpochChief)
    orbitDeputy = Orbit(a,e,i,raan,argP,meanAnomAtEpochDeputy)
    
    N_rChief, N_vChief = orbitChief.cartesian_state_at_time(0)
    N_rDeputy, N_vDeputy = orbitDeputy.cartesian_state_at_time(0)

    # Inertial position of center of mass using cartesian mass avg position 
    # (assuming same mass of chief and deputy)
    N_rCm = (N_rChief + N_rDeputy) / 2.0
    N_vCm = (N_vChief + N_vDeputy) / 2.0
    print(f"rCCM = {N_rCm / 1000.0} [km]")


    # Inertial position vector of the orbit element barycenter
    # This is the average of all the orbit elements and since the only difference
    # between chief and deputy is in true anomaly, we only have to evaluate that element
    trueAnomAvg = (trueAnom + (trueAnom + delTrueAnom)) / 2.0
    meanAnomAvg = true_to_mean_anomaly(trueAnomAvg, e)
    
    orbitAvg = Orbit(a,e,i,raan,argP,meanAnomAvg)

    # Get cartesian state
    N_rAvg, _ = orbitAvg.cartesian_state_at_time(0)

    # Convert to KM
    print(f"rOECM = {N_rAvg / 1000.0} [km]")


    # Compare orbit periods between orbits derived from cartesian avg and orbit element average
    # First define orbit from the cartesian average CM
    orbitCartesianAvg = Orbit.from_cartesian_state(N_rCm, N_vCm)
    periodCartesianAvg = orbitCartesianAvg.period

    periodOeAvg = orbitAvg.period

    periodDiff = np.abs(periodCartesianAvg - periodOeAvg)
    print(f"Difference in periods: {periodDiff} [s]")

if __name__ == "__main__":
    main()