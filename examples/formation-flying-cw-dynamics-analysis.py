"""
In this script, we analyze the difference between using the analytical solution to the CW equations
and the full nonlinear differential equations of relative motion for a leader-follower formation
with a large separation distance
"""
import numpy as np
from spacecraft_dynamics.orbit import Orbit, FormationDynamics

def CwAnaltyicalSolution(A0:float, 
                        B0:float, 
                        alpha:float, 
                        beta:float,
                        x_off:float,
                        y_off:float, 
                        mean_motion:float, 
                        time:float) ->tuple[np.ndarray, np.ndarray]:
    """
    Analytical solution to the CW equations to compute the relative position of a deputy
    (in Hill frame components)
    See eq 14.43 in Analytical Mechanics of Space Systems
    """
    x = A0 * np.cos(mean_motion * time + alpha) + x_off
    y = -2.0 * A0 * np.sin(mean_motion * time + alpha) - (3.0 / 2.0) * mean_motion * time * x_off + y_off
    z = B0 * np.cos(mean_motion * time + beta)
    H_relPosDeputy = np.array([x, y, z])
    
    xDot = -A0 * np.sin(mean_motion * time + alpha) * mean_motion
    yDot = -2.0 * A0 * np.cos(mean_motion * time + alpha) * mean_motion - (3.0 / 2.0) * mean_motion * x_off
    zDot = - B0 * np.sin(mean_motion * time + beta) * mean_motion
    H_relVelDeputy = np.array([xDot, yDot, zDot])
    return H_relPosDeputy, H_relVelDeputy

if __name__== "__main__":

    # Initialize the circular chief orbit
    # NOTE: Only semimajorAxis is specified here, the other parameters are arbitrarily set
    chief_orbit = Orbit(semimajorAxis=7500 * 1000,
                        eccentricity=0.0,
                        inclination=0.0,
                        raan=0.0,
                        argumentOfPeriapsis=0.0,
                        meanAnomalyAtEpoch=0.0,
                        timeOfEpoch=0.0)
    # Define initial condition parameters
    A0 = 0.0
    B0 = 0.0
    x_off = 0.0
    alpha = 0.0
    beta = 0.0
    y_off = 200 * 1000 # [m]
    # Chief orbit mean motion (constant for circular orbits)
    n = chief_orbit.mean_motion

    # Initial relative position and velocity of the deputy according to CW solution
    H_deputyPos0Cw, H_deputyVel0Cw = CwAnaltyicalSolution(A0=A0, B0=B0, alpha=alpha, beta=beta, x_off=x_off, y_off=y_off, mean_motion=n, time=0)
    print(f"Deputy initial relative position according to CW solution: {H_deputyPos0Cw} [m; Hill]")
    print(f"Deputy initial relative velocity according to CW solution: {H_deputyVel0Cw} [m/s; Hill]")

    deputyInitialState = np.hstack([H_deputyPos0Cw, H_deputyVel0Cw])
    deputyStates = [deputyInitialState]

    # Create formation dynamics object
    chiefDynamics = FormationDynamics(chief_orbit, deputyStates)

    # Compute absolute position and velocity of deputy in inertial frame at t=0
    N_position_dep, N_velocity_dep = chief_orbit.deputy_inertial_position_and_velocity_at_time(
                                                    H_deputyPos0Cw, 
                                                    H_deputyVel0Cw, 
                                                    0)
    print(f"Deputy initial absolute position according to CW solution: {N_position_dep} [m; Hill]")
    print(f"Deputy initial absolute velocity according to CW solution: {N_velocity_dep} [m/s; Hill]")

    tInit = chief_orbit.timeOfEpoch
    tMax = 2000
    tStep = 0.1

    # Simulate inertial equations of motion for chief and relative equations of motion for deputy
    solution = chiefDynamics.simulate(t_init=tInit, t_max=tMax, t_step=tStep)

    # Only 1 deputy in this scenario
    deputyIdx = 0
    
    # Get relative position of deputy in Hill frame from simulation of nonlinear relative dynamics [m]
    H_relPosDeputy_final = solution[f'deputy_{deputyIdx}_rho'][-1]
    print(f"Deputy final relative position from simulation of nonlinear relative equations of motion: {H_relPosDeputy_final} [m; Hill]")

    # Compute the separation distance according to the propagation of the nonlinear equations of motion 
    sepDistance = np.linalg.norm(H_relPosDeputy_final)
    print(f"Separation distance from simulation of nonlinear relative equations of motion {sepDistance} [m]")

    # Relative position of deputy according to CW solution
    H_deputyPosFinalCw, H_deputyVelFinalCw = CwAnaltyicalSolution(A0=A0, B0=B0, alpha=alpha, beta=beta, x_off=x_off, y_off=y_off, mean_motion=n, time=tMax)
    sepDistanceCw = np.linalg.norm(H_deputyPosFinalCw)
    print(f"Deputy final relative position from analytical CW solution: {H_deputyPosFinalCw} [m; Hill]")

    # Compute the change in separation distance
    sepDistanceChange = sepDistance - sepDistanceCw
    # sepDistanceChangeInertial = sepDistanceInertial - sepDistanceCw
    print(f"Deputy {deputyIdx} change in separation distance: {sepDistanceChange / 1000} [km]")
