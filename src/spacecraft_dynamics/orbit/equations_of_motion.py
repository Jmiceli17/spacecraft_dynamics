import numpy as np
from spacecraft_dynamics.utils import constants
from spacecraft_dynamics.core.spacecraft import Spacecraft
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from spacecraft_dynamics.orbit.orbit import Orbit

class FormationDynamics():
    # TODO: Make this a sublcass of dynamics class
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
        self.orbit = chiefOrbit
        self.deputy_states = H_deputy_states

    def compute_state_derivatives(self, t: float, state_array: np.ndarray, useLinearizedEoms:bool=False) -> np.ndarray:
        """Compute the state derivatives for the Keplerian dynamics model"""
        
        # Extract states using helper functions
        N_position_chief, N_velocity_chief = self._get_chief_state(state_array)

        # Chief derivatives
        N_position_chief_dot = N_velocity_chief
        r_mag = np.linalg.norm(N_position_chief)
        # Unperturbed 2-body gravity model
        N_velocity_chief_dot = -self.orbit.mu / (r_mag * r_mag * r_mag) * N_position_chief
        
        all_derivatives = [N_position_chief_dot, N_velocity_chief_dot]
        
        # Deputy derivatives
        for i in range(len(self.deputy_states)):
            H_rho_dep, H_rhop_dep = self._get_deputy_state(state_array, i)
            H_rho_dep_dot = H_rhop_dep

            # Calculate deputy relative motion derivatives
            if useLinearizedEoms:
                H_rhop_dep_dot = self._compute_deputy_linearized_accelerations_at_time(t,
                                                                                    H_rho_dep, 
                                                                                    H_rhop_dep,
                                                                                    N_position_chief,
                                                                                    N_velocity_chief)

            else:
                H_rhop_dep_dot = self._compute_deputy_accelerations_at_time(t,
                                                                            H_rho_dep, 
                                                                            H_rhop_dep,
                                                                            N_position_chief,
                                                                            N_velocity_chief)

            all_derivatives.extend([H_rho_dep_dot, H_rhop_dep_dot])

        return np.concatenate(all_derivatives)
    
    def _get_chief_state(self, state_array: np.ndarray) -> tuple:
        """Extract chief state from flat array"""
        return state_array[0:3], state_array[3:6]

    def _get_deputy_state(self, state_array: np.ndarray, deputy_idx: int) -> tuple:
        """Extract deputy state from flat array"""
        start_idx = 6 + deputy_idx * 6
        return state_array[start_idx:start_idx + 3], state_array[start_idx + 3:start_idx + 6]

    def _set_deputy_state(self, state_array: np.ndarray, deputy_idx: int, H_rho: np.ndarray, H_rhop: np.ndarray) -> None:
        """Set deputy state in flat array (for initial conditions)
        NOTE: This modifies the function in place
        """
        start_idx = 6 + deputy_idx * 6
        state_array[start_idx:start_idx + 3] = H_rho
        state_array[start_idx + 3:start_idx + 6] = H_rhop

    def _compute_deputy_accelerations_at_time(self, t, H_rho_dep, H_rhop_dep, N_pos_chief, N_vel_chief)-> np.ndarray:
        """
        The relative equations of motion for a deputy relative to a chief
        NOTE: These are the "exact" relative equations of motion (not the linearized set)
        See eq 14.13 in Analytical Mechanics of Space Systems

        Args:
            t:
            H_rho_dep:
            H_rhop_dep:
            N_pos_chief:
            N_vel_chief:

        Returns: Relative state acceleration vector of xDdot, yDdot, and zDdot
        """
        x = H_rho_dep[0]
        y = H_rho_dep[1]
        z = H_rho_dep[2]

        xDot = H_rhop_dep[0]
        yDot = H_rhop_dep[1]
        zDot = H_rhop_dep[2]

        # Magnitude of position
        rChief = np.linalg.norm(N_pos_chief)

        dcm_HN = self.orbit.hill_frame_at_time(t)
        # Get the r-direction of the Hill frame expressed in inertial coordinates, this is the top row of the [HN] DCM
        N_ohat_r = dcm_HN[0]
        # Get the radial component of velocity rcDot
        rcDot = np.dot(N_vel_chief, N_ohat_r)

        N_position_dep, _ = self.orbit.deputy_inertial_position_and_velocity_at_time(H_rho_dep, 
                                                                                    H_rhop_dep, 
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
    

    def _compute_deputy_linearized_accelerations_at_time(self,  t, H_rho_dep, H_rhop_dep, N_pos_chief, N_vel_chief):
        """
        Compute the linearized relative equations of motion for a deputy relative to a chief
        See eq 14.19 in Analytical Mechanics of Space Systems
        """

        x = H_rho_dep[0]
        y = H_rho_dep[1]
        z = H_rho_dep[2]

        xDot = H_rhop_dep[0]
        yDot = H_rhop_dep[1]
        zDot = H_rhop_dep[2]

        # Magnitude of position
        rChief = np.linalg.norm(N_pos_chief)

        dcm_HN = self.orbit.hill_frame_at_time(t)
        # Get the r-direction of the Hill frame expressed in inertial coordinates, this is the top row of the [HN] DCM
        N_ohat_r = dcm_HN[0]
        # Get the radial component of velocity rcDot
        rcDot = np.dot(N_vel_chief, N_ohat_r)

        true_anom_rate = self.orbit.true_anomaly_rate_at_time(t)

        p = self.orbit.semi_latus_rectum
        
        # Compute state accelerations
        xDdot = 2.0 * true_anom_rate * (yDot - (y * (rcDot / rChief))) + (x * true_anom_rate * true_anom_rate) * (1.0 + (2.0 * (rChief / p)))
        yDdot =  -2.0 * true_anom_rate * (xDot - (x * (rcDot / rChief))) + (y * true_anom_rate * true_anom_rate) * (1.0 - (rChief / p))
        zDdot = -rChief / p * true_anom_rate * true_anom_rate * z
        rhop_dot = np.array([xDdot, yDdot, zDdot])
        return rhop_dot

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



class CwDynamics():
    # TODO: Make this a sublcass of dynamics class
    def __init__(self, chiefOrbit:'Orbit', H_deputy_states:list[np.ndarray]):
        """
        Initialize the Keplerian (i.e. unperturbed, two-body) dynamics model
        This uses the CW equations for modeling the dynamics of the deputies and therefore requires the chief to be in a 
        nearly circular obit
        Args:
            chiefOrbit (Orbit): The chief orbit
            H_deputy_states (list): List of initial Hill frame relative position and velocity vectors each deputy
        """
        self.orbit = chiefOrbit
        if self.orbit.eccentricity > 1e-5:
            raise ValueError("Eccentricity must be less than 1e-5 for CW dynamics")
        self.deputy_states = H_deputy_states

    def _get_chief_state(self, state_array: np.ndarray) -> tuple:
        """Extract chief state from flat array"""
        return state_array[0:3], state_array[3:6]

    def _get_deputy_state(self, state_array: np.ndarray, deputy_idx: int) -> tuple:
        """Extract deputy state from flat array"""
        start_idx = 6 + deputy_idx * 6
        return state_array[start_idx:start_idx + 3], state_array[start_idx + 3:start_idx + 6]


    def compute_state_derivatives(self, t: float, state_array: np.ndarray) -> np.ndarray:
        """Compute the state derivatives for the Keplerian dynamics model"""
        
        # Extract states using helper functions
        N_position_chief, N_velocity_chief = self._get_chief_state(state_array)
        position_chief_mag = np.linalg.norm(N_position_chief)
        velocity_chief_mag = np.linalg.norm(N_velocity_chief)
        # Chief derivatives
        N_position_chief_dot = N_velocity_chief
        r_mag = np.linalg.norm(N_position_chief)
        # Unperturbed 2-body gravity model
        N_velocity_chief_dot = -self.orbit.mu / (r_mag * r_mag * r_mag) * N_position_chief
        
        all_derivatives = [N_position_chief_dot, N_velocity_chief_dot]
        
        # Deputy derivatives
        for i in range(len(self.deputy_states)):
            H_rho_dep, H_rhop_dep = self._get_deputy_state(state_array, i)
            H_rho_dep_dot = H_rhop_dep

            # Calculate deputy relative motion derivatives
            H_rhop_dep_dot = self._compute_deputy_accelerations_at_time(t,
                                                                        H_rho_dep, 
                                                                        H_rhop_dep,
                                                                        N_position_chief,
                                                                        N_velocity_chief)

            all_derivatives.extend([H_rho_dep_dot, H_rhop_dep_dot])

        return np.concatenate(all_derivatives)


    def _compute_deputy_accelerations_at_time(self,  t, H_rho_dep, H_rhop_dep, N_pos_chief, N_vel_chief):
        """
        Compute the unforced CW equations of motion for a deputy relative to a chief
        See eq 14.21 in Analytical Mechanics of Space Systems
        """

        x = H_rho_dep[0]
        y = H_rho_dep[1]
        z = H_rho_dep[2]

        xDot = H_rhop_dep[0]
        yDot = H_rhop_dep[1]
        zDot = H_rhop_dep[2]

        n = self.orbit.mean_motion
        
        # Compute state accelerations
        xDdot = (2.0 * n * yDot) + (3.0 * n * n * x)
        yDdot =  -2.0 * n * xDot 
        zDdot = -n * n * z
        rhop_dot = np.array([xDdot, yDdot, zDdot])
        return rhop_dot

    def simulate(self,
                t_init: float,
                t_max: float,
                t_step: float = 0.1,
                disturbance_acceleration: Callable | None = None) -> dict:
        """
        Simulate the formation dynamics using RK4 integration.

        Args:
            t_init (float): Initial time [s]
            t_max (float): End time [s]
            t_step (float): Time step [s]
            disturbance_acceleration (callable): Optional disturbance function f(t, state)
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

            k1 = t_step * self.compute_state_derivatives(t, current_state)
            k2 = t_step * self.compute_state_derivatives(t + 0.5 * t_step, current_state + 0.5 * k1)
            k3 = t_step * self.compute_state_derivatives(t + 0.5 * t_step, current_state + 0.5 * k2)
            k4 = t_step * self.compute_state_derivatives(t + t_step, current_state + k3)

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