"""
Example use of CW equations of motion to propagate relative state of a deputy
relative to a chief in a circular orbit
"""
import numpy as np
from spacecraft_dynamics.orbit import Orbit, CwDynamics

if __name__== "__main__":

    # Initialize the circular chief orbit
    # NOTE: Only semimajor_axis is specified here, the other parameters are arbitrarily set
    chief_orbit = Orbit(semimajor_axis=6800 * 1000,
                        eccentricity=0.0,
                        inclination=0.0,
                        raan=0.0,
                        argument_of_periapsis=0.0,
                        mean_anomaly_at_epoch=0.0,
                        time_of_epoch=0.0)

    # Initialize the relative position and velocity of the deputy 
    # (in Hill frame coordinates) [m], [m/s]
    H_rel_position_dep = np.array([1.299038,-1.0000,0.3213938]) * 1000
    H_rel_velocity_dep = np.array([-0.000844437,-0.00292521,-0.000431250]) * 1000


    deputyInitialState = np.hstack([H_rel_position_dep, H_rel_velocity_dep])
    deputyStates = [deputyInitialState]

    circularFormationDynamics = CwDynamics(chief_orbit, deputyStates)

    tInit = chief_orbit.time_of_epoch
    useLinearizedEoms = True
    tMax = 1300
    tStep = 0.01
    solution = circularFormationDynamics.simulate(t_init=tInit, t_max=tMax, t_step=tStep)

    # Print chief's final position and velocity
    print(f"======== Simulation complete ========")
    print(f"End time: {solution['time'][-1]}")
    print(f"Final chief position: {solution['position'][-1]} [m]")
    print(f"Final chief velocity: {solution['velocity'][-1]} [m/s]")
    
    # Only 1 deputy in this scenario
    deputy_idx = 0
    H_rho_dep_final = solution[f'deputy_{deputy_idx}_rho'][-1]
    H_rhop_dep_final = solution[f'deputy_{deputy_idx}_rhop'][-1]
    print(f"Deputy relative state converted in Hill frame:")
    print(f"  Final deputy {deputy_idx} rho: {H_rho_dep_final / 1000} [km; Hill]")
    print(f"  Final deputy {deputy_idx} rhop: {H_rhop_dep_final / 1000} [km/s; Hill]")