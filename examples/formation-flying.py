"""
Script that uses the full nonlinear (2 body) equations of motion to model the motion of deputies
relative to a chief and simulates the formation
"""

import numpy as np
from spacecraft_dynamics.orbit import Orbit, FormationDynamics

if __name__== "__main__":

    # Initialize position and velocity of chief [m], [m/s]
    N_position_chief = np.array([-6685.20926,601.51244,3346.06634]) * 1000
    N_velocity_chief = np.array([-1.74294,-6.70242,-2.27739]) * 1000
    chief_orbit = Orbit.from_cartesian_state(N_position_chief, N_velocity_chief)

    # Initialize the relative position and velocity of the deputy 
    # (in Hill frame coordinates) [m], [m/s]
    H_rel_position_dep = np.array([-81.22301,248.14201,94.95904]) * 1000
    H_rel_velocity_dep = np.array([0.47884,0.14857,0.13577]) * 1000

    # Compute absolute position and velocity of deputy in inertial frame at t=0
    N_position_dep, N_velocity_dep = chief_orbit.deputy_inertial_position_and_velocity_at_time(
                                                    H_rel_position_dep, 
                                                    H_rel_velocity_dep, 
                                                    0)
    # Define deputy orbit to compare results of propagating absolute equations of motion to relative 
    # equations of motion
    deputy_orbit = Orbit.from_cartesian_state(N_position_dep, N_velocity_dep)
    deputyDynamics = FormationDynamics(deputy_orbit, [])

    # Convert back to km for printing
    N_position_dep /= 1000
    N_velocity_dep /= 1000
    deputyInitialState = np.hstack([H_rel_position_dep, H_rel_velocity_dep])
    deputyStates = [deputyInitialState]

    formationDynamics = FormationDynamics(chief_orbit, deputyStates)

    print(f"initial_rd_N = [{N_position_dep[0]}, {N_position_dep[1]}, {N_position_dep[2]}] # km")
    print(f"initial_vd_N = [{N_velocity_dep[0]}, {N_velocity_dep[1]}, {N_velocity_dep[2]}] # km/s")

    tInit = chief_orbit.time_of_epoch
    useLinearizedEoms = True
    tMax = 2000
    tStep = 0.01
    solution = formationDynamics.simulate(t_init=tInit, t_max=tMax, t_step=tStep, useLinearizedEoms=useLinearizedEoms)

    # Print chief's final position and velocity
    print(f"======== Simulation complete ========")
    print(f"Used linearized equations of motion? {useLinearizedEoms}")
    print(f"End time: {solution['time'][-1]}")
    print(f"Final chief position:", solution["position"][-1])
    print(f"Final chief velocity:", solution["velocity"][-1])

    # Print each deputy's final state
    num_deputies = len(deputyStates)
    for i in range(num_deputies):
        H_relPosDeputy_final = solution[f'deputy_{i}_rho'][-1]
        H_relVelDeputy_final = solution[f'deputy_{i}_rhop'][-1]
        print(f"Deputy relative state converted in Hill frame:")
        print(f"  Final deputy {i} rho: {H_relPosDeputy_final / 1000} [km; Hill]")
        print(f"  Final deputy {i} rhop: {H_relVelDeputy_final / 1000} [km/s; Hill]")

        N_pos_dep_final, N_vel_dep_final = chief_orbit.deputy_inertial_position_and_velocity_at_time(
                                                    H_relPosDeputy_final, 
                                                    H_relVelDeputy_final, 
                                                    tMax)
        print(f"Deputy absolute state in inertial frame:")
        print(f"  Final deputy {i} N_pos: {N_pos_dep_final / 1000}")
        print(f"  Final deputy {i} N_vel: {N_vel_dep_final / 1000}")

    deputySolution = deputyDynamics.simulate(t_init=tInit,t_max=tMax, t_step=tStep)
    # Print chief's final position and velocity (in this orbit, "chief" actually refers to the deputy)
    print(f"Deputy inertial simulation end time: {deputySolution['time'][-1]}")
    print(f"Final deputy inertial position: {deputySolution['position'][-1] / 1000}" )
    print(f"Final deputy inertial velocity: {deputySolution['velocity'][-1] / 1000}" )