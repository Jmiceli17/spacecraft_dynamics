
import numpy as np
from typing import Optional, Tuple
from ..utils import constants


class Orbit:
    """
    Class to store invariant classical orbital elements.
    
    The classical orbital elements are:
    - semimajor axis (a): [m]
    - eccentricity (e): dimensionless
    - inclination (i): [rad]
    - right ascension of ascending node (RAAN, Ω): [rad]
    - argument of periapsis (ω): [rad]
    - mean anomaly at epoch (M₀): [rad]
    - time of epoch (t₀): [s]
    
    These elements are invariant under two-body dynamics (no perturbations).
    """
    
    def __init__(self, 
                 semimajor_axis: float,
                 eccentricity: float,
                 inclination: float,
                 raan: float,
                 argument_of_periapsis: float,
                 mean_anomaly_at_epoch: float,
                 time_of_epoch: float = 0.0,
                 central_body: str = "Earth"):
        """
        Initialize the Orbit class with classical orbital elements.
        
        Args:
            semimajor_axis (float): Semimajor axis [m]
            eccentricity (float): Eccentricity (dimensionless)
            inclination (float): Inclination [rad]
            raan (float): Right ascension of ascending node [rad]
            argument_of_periapsis (float): Argument of periapsis [rad]
            mean_anomaly_at_epoch (float): Mean anomaly at epoch [rad]
            time_of_epoch (float): Time of epoch [s]
            central_body (str): Central body (default: "Earth")
        """
        self.semimajor_axis = semimajor_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.argument_of_periapsis = argument_of_periapsis
        self.mean_anomaly_at_epoch = mean_anomaly_at_epoch
        self.time_of_epoch = time_of_epoch
        self.central_body = central_body
        
        # Set gravitational parameter based on central body
        if central_body == "Earth":
            self.mu = constants.MU_EARTH_M
        elif central_body == "Mars":
            self.mu = constants.MU_MARS
        else:
            raise ValueError(f"Central body '{central_body}' not supported")
        
        # Validate orbital elements
        self._validate_elements()
    
    def _validate_elements(self):
        """Validate the orbital elements for physical consistency."""
        if self.eccentricity < 0:
            raise ValueError("Eccentricity must be non-negative")
        
        if self.inclination < 0 or self.inclination > np.pi:
            raise ValueError("Inclination must be in range [0, π]")
        
        if self.raan < 0 or self.raan > 2 * np.pi:
            raise ValueError("RAAN must be in range [0, 2π]")
        
        if self.argument_of_periapsis < 0 or self.argument_of_periapsis > 2 * np.pi:
            raise ValueError("Argument of periapsis must be in range [0, 2π]")
        
        if self.mean_anomaly_at_epoch < 0 or self.mean_anomaly_at_epoch > 2 * np.pi:
            raise ValueError("Mean anomaly at epoch must be in range [0, 2π]")
        
        # For elliptical orbits, semimajor axis should be positive
        if self.eccentricity < 1 and self.semimajor_axis <= 0:
            raise ValueError("Semimajor axis must be positive for elliptical orbits")
        
        # For parabolic orbits, semimajor axis should be infinite (represented as 0)
        if self.eccentricity == 1 and self.semimajor_axis != 0:
            raise ValueError("Semimajor axis should be 0 for parabolic orbits")
    
    @property
    def period(self) -> float:
        """Calculate orbital period [s]."""
        if self.eccentricity >= 1:
            return np.inf  # Parabolic or hyperbolic orbits
        return 2 * np.pi * np.sqrt(self.semimajor_axis**3 / self.mu)
    
    @property
    def mean_motion(self) -> float:
        """Calculate mean motion [rad/s]."""
        if self.eccentricity >= 1:
            return 0  # Parabolic or hyperbolic orbits
        return np.sqrt(self.mu / self.semimajor_axis**3)
    
    @property
    def semi_latus_rectum(self) -> float:
        """Calculate semi-latus rectum [m]."""
        if self.eccentricity == 1:
            # For parabolic orbits, use the parameter p
            return self.semimajor_axis  # In this case, semimajor_axis represents p
        return self.semimajor_axis * (1 - self.eccentricity**2)
    
    def mean_anomaly_at_time(self, time: float) -> float:
        """
        Calculate mean anomaly at a given time.
        Args:
            time (float): Time [s]
        Returns:
            float: Mean anomaly at the specified time [rad]
        """
        if self.eccentricity >= 1:
            raise ValueError("Mean anomaly is not defined for parabolic or hyperbolic orbits")
        
        dt = time - self.time_of_epoch
        mean_anomaly = self.mean_anomaly_at_epoch + self.mean_motion * dt
        
        # Normalize to [0, 2π]
        return mean_anomaly % (2 * np.pi)
    
    def eccentric_anomaly_at_time(self, time: float) -> float:
        """
        Calculate eccentric anomaly at a given time using Kepler's equation.
        
        Args:
            time (float): Time [s]
            
        Returns:
            float: Eccentric anomaly at the specified time [rad]
        """
        if self.eccentricity >= 1:
            raise ValueError("Eccentric anomaly is not defined for parabolic or hyperbolic orbits")
        
        mean_anomaly = self.mean_anomaly_at_time(time)
        
        # Solve Kepler's equation: M = E - e*sin(E)
        # Use Newton's method for iterative solution
        E = mean_anomaly  # Initial guess
        
        for _ in range(10):  # Maximum 10 iterations
            f = E - self.eccentricity * np.sin(E) - mean_anomaly
            f_prime = 1 - self.eccentricity * np.cos(E)
            
            if abs(f_prime) < 1e-12:
                break
                
            E_new = E - f / f_prime
            
            if abs(E_new - E) < 1e-12:
                break
                
            E = E_new
        
        return E
    
    def true_anomaly_at_time(self, time: float) -> float:
        """
        Calculate true anomaly at a given time.
        
        Args:
            time (float): Time [s]
            
        Returns:
            float: True anomaly at the specified time [rad]
        """
        if self.eccentricity >= 1:
            raise ValueError("True anomaly calculation not implemented for parabolic/hyperbolic orbits")
        
        E = self.eccentric_anomaly_at_time(time)
        
        # Convert eccentric anomaly to true anomaly
        cos_nu = (np.cos(E) - self.eccentricity) / (1 - self.eccentricity * np.cos(E))
        sin_nu = (np.sin(E) * np.sqrt(1 - self.eccentricity**2)) / (1 - self.eccentricity * np.cos(E))
        
        nu = np.arctan2(sin_nu, cos_nu)
        
        # Ensure positive angle
        if nu < 0:
            nu += 2 * np.pi
            
        return nu

    def true_anomaly_rate_at_time(self, time: float) -> float:
        """
        Compute the true anomaly rate (dnu/dt) at a given time [rad/s].
        Args:
            time (float): Time [s]
        Returns:
            float: True anomaly rate at the specified time [rad/s]
        """
        if self.eccentricity >= 1:
            raise ValueError("True anomaly rate not implemented for parabolic/hyperbolic orbits")

        nu = self.true_anomaly_at_time(time)
        p = self.semi_latus_rectum
        h = np.sqrt(self.mu * p)
        r = p / (1 + self.eccentricity * np.cos(nu))
        return h / (r ** 2)

    def cartesian_state_at_time(self, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Cartesian position and velocity at a given time by propagating true anomaly and converting 
        to cartesian coordinates
        
        Args:
            time (float): Time [s]
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (position, velocity) in inertial frame [m, m/s]
        """
        if self.eccentricity >= 1:
            raise ValueError("Cartesian state calculation not implemented for parabolic/hyperbolic orbits")
        
        # Calculate true anomaly
        nu = self.true_anomaly_at_time(time)
        
        # Calculate radius
        r = self.semi_latus_rectum / (1 + self.eccentricity * np.cos(nu))
        
        # Position in perifocal frame
        P_r = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        
        # Velocity in perifocal frame
        h = np.sqrt(self.mu * self.semi_latus_rectum)
        P_v = np.array([-h * np.sin(nu) / self.semi_latus_rectum, 
                         h * (self.eccentricity + np.cos(nu)) / self.semi_latus_rectum, 
                         0])
        
        # Rotation matrices for coordinate transformation
        R3_raan = np.array([[np.cos(self.raan), -np.sin(self.raan), 0],
                           [np.sin(self.raan), np.cos(self.raan), 0],
                           [0, 0, 1]])
        
        R1_inc = np.array([[1, 0, 0],
                          [0, np.cos(self.inclination), -np.sin(self.inclination)],
                          [0, np.sin(self.inclination), np.cos(self.inclination)]])
        
        R3_argp = np.array([[np.cos(self.argument_of_periapsis), -np.sin(self.argument_of_periapsis), 0],
                           [np.sin(self.argument_of_periapsis), np.cos(self.argument_of_periapsis), 0],
                           [0, 0, 1]])
        
        # Combined rotation matrix (perifocal to inertial)
        dcm_NP = R3_raan @ R1_inc @ R3_argp
        
        # Transform to inertial frame
        N_r = dcm_NP @ P_r
        N_v = dcm_NP @ P_v
        
        return N_r, N_v



    def hill_frame_at_time(self, time:float) -> np.ndarray:
        """
        Get the DCM [HN] that maps the inertial frame to the Hill frame of this orbit
        at the requested time

        Args:
            time (float): Time [s]
        
        Returns:
            np.ndarray: The DCM [HN]
        """

        # Get the cartesian state at this time
        N_r, N_v = self.cartesian_state_at_time(time)
        N_angular_momentum = np.cross(N_r, N_v)

        # Compute basis vectors of Hill frame in inertial frame
        # NOTE: these are all row vectors here, i.e. equivalent to [N_ox]^T
        N_hhat_r = N_r / np.linalg.norm(N_r)
        N_hhat_h = N_angular_momentum / np.linalg.norm(N_angular_momentum)
        N_hhat_theta = np.cross(N_hhat_h, N_hhat_r)

        # The basis vectors for the rows of the DCM
        dcm_HN = np.array([N_hhat_r, N_hhat_theta, N_hhat_h])
        return dcm_HN


    def orbit_angular_velocity_at_time(self, time:float) -> np.ndarray:
        """
        Compute the angular velocity of the orbit/Hill frame relative to the inertial frame
        at a given time [rad/s]

        Args:
            time (float): Time [s]

        Returns:
            np.ndarray: Angular velocity of the Hill frame relative to the inertial frame at 
            this time, expressed in inertial frame coordinates [rad/s]
        """
        # Get the cartesian state at this time
        N_r, N_v = self.cartesian_state_at_time(time)
        N_angular_momentum = np.cross(N_r, N_v)

        # Angular momentum direction vector
        h_mag = np.linalg.norm(N_angular_momentum)
        N_hhat_h = N_angular_momentum / h_mag

        # Instantaneous true anomaly rate
        r_magnitude = np.linalg.norm(N_r)
        nu_dot = h_mag / (r_magnitude * r_magnitude)

        N_omega_HN = nu_dot * N_hhat_h
        return N_omega_HN

    def deputy_inertial_position_and_velocity_at_time(self, H_rho:np.ndarray, H_rhop:np.ndarray, time:float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert relative position of deputy in Hill frame to absolute position
        of deputy in inertial frame at a given time
        """
        N_r_chief, N_v_chief = self.cartesian_state_at_time(time)
        dcm_HN = self.hill_frame_at_time(time)
        N_omega_HN = self.orbit_angular_velocity_at_time(time)

        # Inertial frame relative position and velocity [m], [m/s]
        dcm_NH = dcm_HN.T
        N_rho_deputy = dcm_NH @ H_rho
        N_rhop_deputy = dcm_NH @ H_rhop

        # Deputy intertial position and velocity [m], [m/s]
        N_r_deputy = N_rho_deputy + N_r_chief
        N_v_deputy = N_rhop_deputy + np.cross(N_omega_HN, N_rho_deputy) + N_v_chief
        return N_r_deputy, N_v_deputy


    @classmethod
    def from_cartesian_state(cls, 
                           position: np.ndarray, 
                           velocity: np.ndarray, 
                           time: float = 0.0,
                           central_body: str = "Earth") -> 'Orbit':
        """
        Create an Orbit object from Cartesian state.
        
        Args:
            position (np.ndarray): Position vector in inertial frame [m]
            velocity (np.ndarray): Velocity vector in inertial frame [m/s]
            time (float): Time corresponding to the state [s]
            central_body (str): Central body (default: "Earth")
            
        Returns:
            Orbit: Orbit object with classical orbital elements
        """
        # Set gravitational parameter
        if central_body == "Earth":
            mu = constants.MU_EARTH_M
        elif central_body == "Mars":
            mu = constants.MU_MARS
        else:
            raise ValueError(f"Central body '{central_body}' not supported")
        
        # Calculate orbital elements using the same algorithm as in KeplerianDynamics
        r_mag = np.linalg.norm(position)
        v_mag = np.linalg.norm(velocity)
        
        h = np.cross(position, velocity)
        h_mag = np.linalg.norm(h)
        
        K = np.array([0, 0, 1])
        n = np.cross(K, h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - mu / r_mag) * position - np.dot(position, velocity) * velocity) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Specific energy
        eta = v_mag**2 / 2 - mu / r_mag
        
        # Semimajor axis
        if ecc != 1.0:
            sma = -mu / (2 * eta)
        else:
            sma = 0  # Parabolic orbit
        
        # Inclination
        # if h[2] == 0.0:
        #     inc = 0.0
        # else:
        #     inc = np.arccos(h[2] / h_mag)
        cos_inc = np.clip(h[2] / h_mag, -1.0, 1.0)
        inc = np.arccos(cos_inc)

        # RAAN
        # if n[0] == 0.0:
        #     raan = 0.0
        # else:
        #     raan = np.arccos(n[0] / n_mag)
        # if n[1] < 0:
        #      raan = 2 * np.pi - raan
        if n_mag < 1e-12:
            # Near equatorial orbit
            raan = 0.0
        else:
            cos_raan = np.clip(n[0] / n_mag, -1.0, 1.0)
            raan = np.arccos(cos_raan)
            if n[1] < 0:
                raan = 2 * np.pi - raan
        
        # Argument of periapsis
        # if np.dot(n, e_vec) == 0.0:
        #     argp = 0.0
        # else:
        #     argp = np.arccos(np.dot(n, e_vec) / (n_mag * ecc))
        # if e_vec[2] < 0.0:
        #     argp = 2 * np.pi - argp
        if n_mag < 1e-12 or ecc < 1e-12:
            # Near equatorial or circular orbit
            argp = 0.0
        else:
            cos_argp = np.clip(np.dot(n, e_vec) / (n_mag * ecc), -1.0, 1.0)
            argp = np.arccos(cos_argp)
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp

        # True anomaly
        # print(f"np.dot(e_vec, position): {np.dot(e_vec, position)}")
        # if np.dot(e_vec, position) == 0.0:
        #     nu = 0.0
        # else:
        #     print(f"e_vec: {e_vec}")
        #     print(f"position: {position}")
        #     print(f"ecc: {ecc}")
        #     print(f"r_mag: {r_mag}")
        #     print(f"np.dot(e_vec, position) / (ecc * r_mag): {np.dot(e_vec, position) / (ecc * r_mag)}")
        #     nu = np.arccos(np.dot(e_vec, position) / (ecc * r_mag))
        # if np.dot(position, velocity) < 0:
        #     nu = 2 * np.pi - nu
        if ecc < 1e-12:
            # Circular orbit
            nu = 0.0
        else:
            cos_nu = np.clip(np.dot(e_vec, position) / (ecc * r_mag), -1.0, 1.0)
            nu = np.arccos(cos_nu)
            if np.dot(position, velocity) < 0:
                nu = 2 * np.pi - nu

        # Calculate eccentric anomaly
        if ecc < 1:
            sqrt_term = np.sqrt((1 - ecc) / (1 + ecc))
            tan_half_theta = np.tan(nu / 2)
            tan_half_E = sqrt_term * tan_half_theta
    
            # Compute eccentric anomaly E = 2 * arctan(tan(E/2))
            E = 2 * np.arctan(tan_half_E)
            E = E % (2 * np.pi)
            # cos_E = (ecc + np.cos(nu)) / (1 + ecc * np.cos(nu))
            # sin_E = (np.sin(nu) * np.sqrt(1 - ecc**2)) / (1 + ecc * np.cos(nu))
            # E = np.arctan2(sin_E, cos_E)
            
            # Mean anomaly
            M = E - (ecc * np.sin(E))
        else:
            # For parabolic orbits, use a different approach
            M = 0  # Placeholder - would need special handling for parabolic orbits
        
        return cls(semimajor_axis=sma,
                  eccentricity=ecc,
                  inclination=inc,
                  raan=raan,
                  argument_of_periapsis=argp,
                  mean_anomaly_at_epoch=M,
                  time_of_epoch=time,
                  central_body=central_body)
    
    def __str__(self) -> str:
        """String representation of the orbit."""
        return (f"Orbit(central_body={self.central_body}, "
                f"a={self.semimajor_axis:.2e} m, "
                f"e={self.eccentricity:.6f}, "
                f"i={np.degrees(self.inclination):.2f}°, "
                f"Ω={np.degrees(self.raan):.2f}°, "
                f"ω={np.degrees(self.argument_of_periapsis):.2f}°, "
                f"M₀={np.degrees(self.mean_anomaly_at_epoch):.2f}°, "
                f"t₀={self.time_of_epoch:.1f} s)")
    
    def __repr__(self) -> str:
        """Detailed string representation of the orbit."""
        return (f"Orbit(\n"
                f"  semimajor_axis={self.semimajor_axis},\n"
                f"  eccentricity={self.eccentricity},\n"
                f"  inclination={self.inclination},\n"
                f"  raan={self.raan},\n"
                f"  argument_of_periapsis={self.argument_of_periapsis},\n"
                f"  mean_anomaly_at_epoch={self.mean_anomaly_at_epoch},\n"
                f"  time_of_epoch={self.time_of_epoch},\n"
                f"  central_body='{self.central_body}'\n"
                f")")