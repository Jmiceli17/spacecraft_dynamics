import numpy as np

from spacecraft_dynamics.orbit import Orbit


if __name__ == "__main__":

    """
    Use the following inertial position and velocity for the chief in both scenarios
    """
    # Chief inertial position and velocity [m], [m/s]
    N_r_chief = np.array([-4893.268, 3864.478, 3169.646]) * 1000.0
    N_v_chief = np.array([-3.91386, -6.257673, 1.59797]) * 1000.0
    chief_orbit = Orbit.from_cartesian_state(N_r_chief, N_v_chief)
    print(f"Chief orbit:\n{chief_orbit}")


    """
    CASE 1:
    Given inertial position and velocity of a the chief and deputy spacecraft, 
    determine the coresponing rho and rho' vector in the Chief's orbit frame
    """

    # Deputy intertial position and velocity 
    N_r_deputy = np.array([-4892.98, 3863.073, 3170.619]) * 1000.0
    N_v_deputy = np.array([-3.913302, -6.258661, 1.598199]) * 1000.0

    # Relative position and velocity of the deputy in the Chief's Hill frame [m], [m/s]
    N_rho_deputy = N_r_deputy - N_r_chief
    dcm_HN = chief_orbit.hill_frame_at_time(0)
    H_rho_deputy = dcm_HN @ N_rho_deputy

    # Orbit frame angular velocity relative to inertial frame
    N_omega_HN = chief_orbit.orbit_angular_velocity_at_time(0)
    H_omega_HN = dcm_HN @ N_omega_HN

    H_rhop_deputy = dcm_HN @ ((N_v_deputy - N_v_chief) - (np.cross(N_omega_HN, N_rho_deputy)))

    # Convert to [km], [km/s]
    H_rho_deputy = H_rho_deputy / 1000.0
    H_rhop_deputy = H_rhop_deputy / 1000.0
    print(f"H_rhop_deputy: {H_rhop_deputy}")
    print(f"Case 1:")
    print(f"rho_H = [{H_rho_deputy[0]}, {H_rho_deputy[1]}, {H_rho_deputy[2]}]")
    print(f"rhoP_H = [{H_rhop_deputy[0]}, {H_rhop_deputy[1]}, {H_rhop_deputy[2]}]")





    """
    CASE 2:
    Given inertial position and velocity of a the chief and the Hill frame relative position
    of the deputy, determine the corresponding absolute inertial position and velocity
    of the deputy spacecraft
    """
    # Hill frame relative position and velocity of the deputy [m], [m/s]
    H_rho_deputy = np.array([-0.537,1.221,1.106]) * 1000
    H_rhop_deputy = np.array([0.000486,0.001158,0.0005590]) * 1000

    # Inertial frame relative position and velocity [m], [m/s]
    dcm_NH = dcm_HN.T
    N_rho_deputy = dcm_NH @ H_rho_deputy
    N_rhop_deputy = dcm_NH @ H_rhop_deputy

    # Deputy intertial position and velocity [m], [m/s]
    N_r_deputy = N_rho_deputy + N_r_chief
    N_v_deputy = N_rhop_deputy + np.cross(N_omega_HN, N_rho_deputy) + N_v_chief

    # Convert to [km], [km/s]
    N_r_deputy = N_r_deputy / 1000
    N_v_deputy = N_v_deputy / 1000

    print(f"Case 2:")
    print(f"rd_N = [{N_r_deputy[0]}, {N_r_deputy[1]}, {N_r_deputy[2]}]")
    print(f"vd_N = [{N_v_deputy[0]}, {N_v_deputy[1]}, {N_v_deputy[2]}]")
