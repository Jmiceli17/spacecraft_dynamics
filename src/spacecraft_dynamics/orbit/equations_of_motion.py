import numpy as np
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from spacecraft_dynamics.orbit.orbit import Orbit

class FormationDynamicsBase():
    def __init__(self, chiefOrbit:'Orbit', H_deputy_states:list[np.ndarray]):
        """
        Base class for formation dynamics

        Args:
            chiefOrbit (Orbit): The chief orbit
            H_deputy_states (list): List of initial Hill frame relative position and velocity vectors each deputy
        """
        self.orbit = chiefOrbit
        self.deputy_states = H_deputy_states

    def _get_chief_state(self, stateArray: np.ndarray) -> tuple:
        """
        Extract chief state from flat array

        Args:
            stateArray (ndarray): State array of the formation

        Returns:
            Inertial position and velocity of the chief [m], [m/s]
        """
        return stateArray[0:3], stateArray[3:6]

    def _get_deputy_state(self, stateArray: np.ndarray, deputyIdx: int) -> tuple:
        """
        Extract deputy state from flat array

        Args:
            stateArray (ndarray): State array of the formation
            deputyIdx (int): The index of the deputy of interest

        Returns:
            Hill frame relative position and velocity of the deputy [m], [m/s]
        """
        start_idx = 6 + deputyIdx * 6
        return stateArray[start_idx:start_idx + 3], stateArray[start_idx + 3:start_idx + 6]

    def _set_deputy_state(self, stateArray: np.ndarray, deputyIdx: int, H_rho: np.ndarray, H_rhop: np.ndarray) -> None:
        """Set deputy state in flat array (for initial conditions)
        NOTE: This modifies the stateArray in place
        """
        start_idx = 6 + deputyIdx * 6
        stateArray[start_idx:start_idx + 3] = H_rho
        stateArray[start_idx + 3:start_idx + 6] = H_rhop


    def compute_state_derivatives(self, t: float, stateArray: np.ndarray, useLinearizedEoms:bool=False) -> np.ndarray:
        """
        Compute the state derivatives for the dynamics model
            Args:
            t (float): Time [s]
            stateArray (ndarray): State array of the formation, assumes the form [chief pos, chief vel]
            useLinearizedEoms: Flag indicating that linearized equations of motion should be used (not relevantfor all dynamics models)
        """
        raise NotImplementedError

    def simulate(self,
                t_init: float,
                t_max: float,
                t_step: float = 0.1,
                disturbance_acceleration: Callable | None = None,
                useLinearizedEoms:bool=False) -> dict:
        """
        Simulate the formation dynamics using RK4 integration.

        Args:
            t_init (float): Initial time [s]
            t_max (float): End time [s]
            t_step (float): Time step [s]
            disturbance_acceleration (callable): Optional disturbance function f(t, state)
            useLinearizedEoms (bool): Optional flag that if `true` indicates that the linearized relative EOMs
            should be used, otherwise the exact relative EOMs are used
        Returns:
            dict: Dictionary mapping variable names to their time histories
        """
        # 1. Pack initial state: chief inertial + all deputies' Hill-frame states
        chief_pos, chief_vel = self.orbit.cartesian_state_at_time(t_init)
        # Each element in self.deputy_states is a 6-element array: [H_rho, H_rhop]
        initial_state = np.hstack([chief_pos, chief_vel] + [dep for dep in self.deputy_states])

        # Initialize time and state
        t = t_init
        # Total state vector is inertial position/velocity of chief and 
        # Hill frame relative position and velocity of each deputy 
        # [N_rc, N_vc, H_rho_d1, H_rhop_d1, H_rho_d2, H_rhop_d2, ...]
        current_state = initial_state.copy()

        # Initialize solution dictionary
        solution_dict = {
            "time": [t],
            "position": [chief_pos],
            "velocity": [chief_vel],
        }
        num_deputies = len(self.deputy_states)
        for i in range(num_deputies):
            solution_dict[f"deputy_{i}_rho"] = [self.deputy_states[i][:3]]
            solution_dict[f"deputy_{i}_rhop"] = [self.deputy_states[i][3:]]

        # 4. Integration loop
        while t < t_max:

            k1 = t_step * self.compute_state_derivatives(t, current_state, useLinearizedEoms)
            k2 = t_step * self.compute_state_derivatives(t + 0.5 * t_step, current_state + 0.5 * k1, useLinearizedEoms)
            k3 = t_step * self.compute_state_derivatives(t + 0.5 * t_step, current_state + 0.5 * k2, useLinearizedEoms)
            k4 = t_step * self.compute_state_derivatives(t + t_step, current_state + k3, useLinearizedEoms)

            next_state = current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

            # Advance time and state
            t += t_step
            current_state = next_state

            # Unpack and store results
            chief_pos = current_state[0:3]
            chief_vel = current_state[3:6]
            solution_dict["time"].append(t)
            solution_dict["position"].append(chief_pos)
            solution_dict["velocity"].append(chief_vel)
            for i in range(num_deputies):
                start = 6 + i * 6
                solution_dict[f"deputy_{i}_rho"].append(current_state[start:start+3])
                solution_dict[f"deputy_{i}_rhop"].append(current_state[start+3:start+6])

        # Convert lists to arrays for easier use
        for key in solution_dict:
            solution_dict[key] = np.array(solution_dict[key])

        return solution_dict


class FormationDynamics(FormationDynamicsBase):
    # TODO: Add ability to create dynamics from orbit definition of deputies
    def __init__(self, chiefOrbit:'Orbit', H_deputy_states:list[np.ndarray]):
        """
        Initialize the Keplerian (i.e. unperturbed, two-body) dynamics model
        This uses the full non-linear equations of relative motion for the deputies. Optionally, 
        the linearized equations of motion can be used when simulating the dynamics of the formation.

        Args:
            chiefOrbit (Orbit): The chief orbit
            H_deputy_states (list): List of initial Hill frame relative position and velocity vectors each deputy
        """
        super().__init__(chiefOrbit=chiefOrbit, H_deputy_states=H_deputy_states)


    def compute_state_derivatives(self, t: float, stateArray: np.ndarray, useLinearizedEoms:bool=False) -> np.ndarray:
        """
        Compute the state derivatives for the Keplerian dynamics model

        Args:
            t (float): Time [s]
            stateArray (ndarray): State array of the formation, assumes the form [chief pos, chief vel]
            useLinearizedEoms (bool): Flag indicating if linearized equations should be used or not

        Returns:
            Array of state derivatives
        """

        # Extract states using helper functions
        N_position_chief, N_velocity_chief = self._get_chief_state(stateArray)

        # Chief derivatives
        N_position_chief_dot = N_velocity_chief
        r_mag = np.linalg.norm(N_position_chief)
        # Unperturbed 2-body gravity model
        N_velocity_chief_dot = -self.orbit.mu / (r_mag * r_mag * r_mag) * N_position_chief
        
        all_derivatives = [N_position_chief_dot, N_velocity_chief_dot]
        
        # Deputy derivatives
        for i in range(len(self.deputy_states)):
            H_relPosDeputy, H_relVelDeputy = self._get_deputy_state(stateArray, i)
            H_relPosDeputy_dot = H_relVelDeputy

            # Calculate deputy relative motion derivatives
            if useLinearizedEoms:
                H_relVelDeputy_dot = self._compute_deputy_linearized_accelerations_at_time(t,
                                                                                    H_relPosDeputy, 
                                                                                    H_relVelDeputy,
                                                                                    N_position_chief,
                                                                                    N_velocity_chief)

            else:
                H_relVelDeputy_dot = self._compute_deputy_accelerations_at_time(t,
                                                                            H_relPosDeputy, 
                                                                            H_relVelDeputy,
                                                                            N_position_chief,
                                                                            N_velocity_chief)

            all_derivatives.extend([H_relPosDeputy_dot, H_relVelDeputy_dot])

        return np.concatenate(all_derivatives)

    def _compute_deputy_accelerations_at_time(self, t, H_relPosDeputy, H_relVelDeputy, N_posChief, N_velChief)-> np.ndarray:
        """
        The relative equations of motion for a deputy relative to a chief
        NOTE: These are the "exact" relative equations of motion (not the linearized set)
        See eq 14.13 in Analytical Mechanics of Space Systems

        Args:
            t (float): Time [s]
            H_relPosDeputy (ndarray): Hill frame relative position of deputy [m]
            H_relVelDeputy (ndarray): Hill frame relative velocity of deputy [m/s]
            N_posChief (ndarray): Inertial position of chief [m]
            N_velChief (ndarray): Inertial velocity of chief [m/s]

        Returns: Relative state acceleration vector of xDdot, yDdot, and zDdot
        """
        x = H_relPosDeputy[0]
        y = H_relPosDeputy[1]
        z = H_relPosDeputy[2]

        xDot = H_relVelDeputy[0]
        yDot = H_relVelDeputy[1]
        zDot = H_relVelDeputy[2]

        # Magnitude of position
        rChief = np.linalg.norm(N_posChief)

        dcm_HN = self.orbit.hill_frame_at_time(t)
        # Get the r-direction of the Hill frame expressed in inertial coordinates, this is the top row of the [HN] DCM
        N_ohat_r = dcm_HN[0]
        # Get the radial component of velocity rcDot
        rcDot = np.dot(N_velChief, N_ohat_r)

        N_position_dep, _ = self.orbit.deputy_inertial_position_and_velocity_at_time(H_relPosDeputy, 
                                                                                    H_relVelDeputy, 
                                                                                    t)
        # Magnitude of deputy absolute position vector
        r_dep = np.linalg.norm(N_position_dep)

        true_anom_rate = self.orbit.true_anomaly_rate_at_time(t)

        xDdot = 2.0 * true_anom_rate * (yDot - (y * (rcDot / rChief))) + (x * true_anom_rate * true_anom_rate) \
                + self.orbit.mu / (rChief * rChief) - self.orbit.mu / (r_dep * r_dep * r_dep) * (rChief + x)
        
        yDdot = -2.0 * true_anom_rate * (xDot - (x * (rcDot / rChief))) + (y * true_anom_rate * true_anom_rate) \
                - self.orbit.mu / (r_dep * r_dep * r_dep) * y
        
        zDdot = -self.orbit.mu / (r_dep * r_dep * r_dep) * z

        rhop_dot = np.array([xDdot, yDdot, zDdot])
        return rhop_dot

    def _compute_deputy_linearized_accelerations_at_time(self, t, H_relPosDeputy, H_relVelDeputy, N_posChief, N_velChief):
        """
        Compute the linearized relative equations of motion for a deputy relative to a chief
        See eq 14.19 in Analytical Mechanics of Space Systems

        Args:
            t (float): Time [s]
            H_relPosDeputy (ndarray): Hill frame relative position of deputy [m]
            H_relVelDeputy (ndarray): Hill frame relative velocity of deputy [m/s]
            N_posChief (ndarray): Inertial position of chief [m]
            N_velChief (ndarray): Inertial velocity of chief [m/s]

        Returns: Relative state acceleration vector of xDdot, yDdot, and zDdot
        """

        x = H_relPosDeputy[0]
        y = H_relPosDeputy[1]
        z = H_relPosDeputy[2]

        xDot = H_relVelDeputy[0]
        yDot = H_relVelDeputy[1]
        zDot = H_relVelDeputy[2]

        # Magnitude of position
        rChief = np.linalg.norm(N_posChief)

        dcm_HN = self.orbit.hill_frame_at_time(t)
        # Get the r-direction of the Hill frame expressed in inertial coordinates, this is the top row of the [HN] DCM
        N_ohat_r = dcm_HN[0]
        # Get the radial component of velocity rcDot
        rcDot = np.dot(N_velChief, N_ohat_r)

        true_anom_rate = self.orbit.true_anomaly_rate_at_time(t)

        p = self.orbit.semi_latus_rectum
        
        # Compute state accelerations
        xDdot = 2.0 * true_anom_rate * (yDot - (y * (rcDot / rChief))) + (x * true_anom_rate * true_anom_rate) * (1.0 + (2.0 * (rChief / p)))
        yDdot =  -2.0 * true_anom_rate * (xDot - (x * (rcDot / rChief))) + (y * true_anom_rate * true_anom_rate) * (1.0 - (rChief / p))
        zDdot = -rChief / p * true_anom_rate * true_anom_rate * z
        rhop_dot = np.array([xDdot, yDdot, zDdot])
        return rhop_dot



class CwDynamics(FormationDynamicsBase):
    def __init__(self, chiefOrbit:'Orbit', H_deputy_states:list[np.ndarray]):
        """
        Initialize the Keplerian (i.e. unperturbed, two-body) dynamics model
        This uses the CW equations for modeling the dynamics of the deputies and therefore requires the chief to be in a 
        nearly circular obit

        Args:
            chiefOrbit (Orbit): The chief orbit
            H_deputy_states (list): List of initial Hill frame relative position and velocity vectors each deputy
        """
        super().__init__(chiefOrbit=chiefOrbit, H_deputy_states=H_deputy_states)
        if self.orbit.eccentricity > 1e-5:
            raise ValueError("Eccentricity must be less than 1e-5 for CW dynamics")

    def compute_state_derivatives(self, t: float, stateArray: np.ndarray, useLinearizedEoms:bool=False) -> np.ndarray:
        """
        Compute the state derivatives for the Keplerian dynamics model
        
        Args:
            t (float): Time [s]
            stateArray (ndarray): State array of the formation, assumes the form [chief pos, chief vel]
            useLinearizedEoms (bool): Not used

        Returns:
            Array of state derivatives
        """
        
        # Extract states using helper functions
        N_position_chief, N_velocity_chief = self._get_chief_state(stateArray)

        # Chief derivatives
        N_position_chief_dot = N_velocity_chief
        r_mag = np.linalg.norm(N_position_chief)
        # Unperturbed 2-body gravity model
        N_velocity_chief_dot = -self.orbit.mu / (r_mag * r_mag * r_mag) * N_position_chief
        
        all_derivatives = [N_position_chief_dot, N_velocity_chief_dot]
        
        # Deputy derivatives
        for i in range(len(self.deputy_states)):
            H_relPosDeputy, H_relVelDeputy = self._get_deputy_state(stateArray, i)
            H_relPosDeputy_dot = H_relVelDeputy

            # Calculate deputy relative motion derivatives
            H_relVelDeputy_dot = self._compute_deputy_accelerations_at_time(t,
                                                                        H_relPosDeputy, 
                                                                        H_relVelDeputy)

            all_derivatives.extend([H_relPosDeputy_dot, H_relVelDeputy_dot])

        return np.concatenate(all_derivatives)


    def _compute_deputy_accelerations_at_time(self, t, H_relPosDeputy, H_relVelDeputy) -> np.ndarray:
        """
        Compute the unforced CW equations of motion for a deputy relative to a chief
        See eq 14.21 in Analytical Mechanics of Space Systems

        Args:
            t (float): Time [s]
            H_relPosDeputy (ndarray): Relative position of deputy in Chief's Hill frame [m]
            H_relVelDeputy (ndarray): Relative velocity of the deputy in Cheif's Hill frame [m/s]

        Returns:
            Hill frame relative acceleration of the deputy [m/s^2]
        """

        x = H_relPosDeputy[0]
        y = H_relPosDeputy[1]
        z = H_relPosDeputy[2]

        xDot = H_relVelDeputy[0]
        yDot = H_relVelDeputy[1]
        zDot = H_relVelDeputy[2]

        n = self.orbit.mean_motion
        
        # Compute state accelerations
        xDdot = (2.0 * n * yDot) + (3.0 * n * n * x)
        yDdot =  -2.0 * n * xDot 
        zDdot = -n * n * z
        rhop_dot = np.array([xDdot, yDdot, zDdot])
        return rhop_dot
