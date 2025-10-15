
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
                 semimajorAxis: float,
                 eccentricity: float,
                 inclination: float,
                 raan: float,
                 argumentOfPeriapsis: float,
                 meanAnomalyAtEpoch: float,
                 timeOfEpoch: float = 0.0,
                 centralBody: str = "Earth"):
        """
        Initialize the Orbit class with classical orbital elements.
        
        Args:
            semimajorAxis (float): Semimajor axis [m]
            eccentricity (float): Eccentricity (dimensionless)
            inclination (float): Inclination [rad]
            raan (float): Right ascension of ascending node [rad]
            argumentOfPeriapsis (float): Argument of periapsis [rad]
            meanAnomalyAtEpoch (float): Mean anomaly at epoch [rad]
            timeOfEpoch (float): Time of epoch [s]
            centralBody (str): Central body (default: "Earth")
        """
        self.semimajorAxis = semimajorAxis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.argumentOfPeriapsis = argumentOfPeriapsis
        self.meanAnomalyAtEpoch = meanAnomalyAtEpoch
        self.timeOfEpoch = timeOfEpoch
        self.centralBody = centralBody
        
        # Set gravitational parameter based on central body
        if centralBody == "Earth":
            self.mu = constants.MU_EARTH_M
        elif centralBody == "Mars":
            self.mu = constants.MU_MARS
        else:
            raise ValueError(f"Central body '{centralBody}' not supported")
        
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
        
        if self.argumentOfPeriapsis < 0 or self.argumentOfPeriapsis > 2 * np.pi:
            raise ValueError("Argument of periapsis must be in range [0, 2π]")
        
        if self.meanAnomalyAtEpoch < 0 or self.meanAnomalyAtEpoch > 2 * np.pi:
            raise ValueError("Mean anomaly at epoch must be in range [0, 2π]")
        
        # For elliptical orbits, semimajor axis should be positive
        if self.eccentricity < 1 and self.semimajorAxis <= 0:
            raise ValueError("Semimajor axis must be positive for elliptical orbits")
        
        # For parabolic orbits, semimajor axis should be infinite (represented as 0)
        if self.eccentricity == 1 and self.semimajorAxis != 0:
            raise ValueError("Semimajor axis should be 0 for parabolic orbits")
    
    @property
    def period(self) -> float:
        """Calculate orbital period [s]."""
        if self.eccentricity >= 1:
            return np.inf  # Parabolic or hyperbolic orbits
        return 2 * np.pi * np.sqrt(self.semimajorAxis**3 / self.mu)
    
    @property
    def mean_motion(self) -> float:
        """Calculate mean motion [rad/s]."""
        if self.eccentricity >= 1:
            return 0  # Parabolic or hyperbolic orbits
        return np.sqrt(self.mu / self.semimajorAxis**3)
    
    @property
    def semiLatusRectum(self) -> float:
        """Calculate semi-latus rectum [m]."""
        if self.eccentricity == 1:
            # For parabolic orbits, use the parameter p
            return self.semimajorAxis  # In this case, semimajorAxis represents p
        return self.semimajorAxis * (1 - self.eccentricity**2)
    
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
        
        dt = time - self.timeOfEpoch
        mean_anomaly = self.meanAnomalyAtEpoch + self.mean_motion * dt
        
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
        p = self.semiLatusRectum
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
        r = self.semiLatusRectum / (1 + self.eccentricity * np.cos(nu))
        
        # Position in perifocal frame
        P_r = np.array([r * np.cos(nu), r * np.sin(nu), 0])
        
        # Velocity in perifocal frame
        h = np.sqrt(self.mu * self.semiLatusRectum)
        P_v = np.array([-h * np.sin(nu) / self.semiLatusRectum, 
                         h * (self.eccentricity + np.cos(nu)) / self.semiLatusRectum, 
                         0])
        
        # Rotation matrices for coordinate transformation
        R3_raan = np.array([[np.cos(self.raan), -np.sin(self.raan), 0],
                           [np.sin(self.raan), np.cos(self.raan), 0],
                           [0, 0, 1]])
        
        R1_inc = np.array([[1, 0, 0],
                          [0, np.cos(self.inclination), -np.sin(self.inclination)],
                          [0, np.sin(self.inclination), np.cos(self.inclination)]])
        
        R3_argp = np.array([[np.cos(self.argumentOfPeriapsis), -np.sin(self.argumentOfPeriapsis), 0],
                           [np.sin(self.argumentOfPeriapsis), np.cos(self.argumentOfPeriapsis), 0],
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
        N_angularMomentum = np.cross(N_r, N_v)

        # Compute basis vectors of Hill frame in inertial frame
        # NOTE: these are all row vectors here, i.e. equivalent to [N_ox]^T
        N_hhat_r = N_r / np.linalg.norm(N_r)
        N_hhat_h = N_angularMomentum / np.linalg.norm(N_angularMomentum)
        N_hhat_theta = np.cross(N_hhat_h, N_hhat_r)

        # The basis vectors for the rows of the DCM
        dcm_HN = np.array([N_hhat_r, N_hhat_theta, N_hhat_h])
        return dcm_HN


    def orbit_angular_velocity_at_time(self, 
                                    time:float
                                    ) -> np.ndarray:
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
        N_angularMomentum = np.cross(N_r, N_v)

        # Angular momentum direction vector
        angularMomentumMagnitude = np.linalg.norm(N_angularMomentum)
        N_hhat_h = N_angularMomentum / angularMomentumMagnitude

        # Instantaneous true anomaly rate
        rMagnitude = np.linalg.norm(N_r)
        nuDot = angularMomentumMagnitude / (rMagnitude * rMagnitude)

        N_omega_HN = nuDot * N_hhat_h
        return N_omega_HN

    def deputy_inertial_position_and_velocity_at_time(self, 
                                                    H_relPosDeputy:np.ndarray, 
                                                    H_relVelDeputy:np.ndarray, 
                                                    time:float
                                                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: Rename to deputy_inertial_state_at_time
        Convert relative position of deputy in Hill frame to absolute position
        of deputy in inertial frame at a given time
        """
        N_r_chief, N_v_chief = self.cartesian_state_at_time(time)
        dcm_HN = self.hill_frame_at_time(time)
        N_omega_HN = self.orbit_angular_velocity_at_time(time)

        # Inertial frame relative position and velocity [m], [m/s]
        dcm_NH = dcm_HN.T
        N_relPosDeputy = dcm_NH @ H_relPosDeputy
        N_relVelDeputy = dcm_NH @ H_relVelDeputy

        # Deputy intertial position and velocity [m], [m/s]
        N_r_deputy = N_relPosDeputy + N_r_chief
        N_v_deputy = N_relVelDeputy + (np.cross(N_omega_HN, N_relPosDeputy) + N_v_chief)
        return N_r_deputy, N_v_deputy


    def deputy_hill_frame_state_at_time(self,
                                        N_r:np.ndarray,
                                        N_rDot:np.ndarray,
                                        time:float
                                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert absolute inertial position and velocity of a deputy into 
        Hill frame relative coordinates
        Args:
            N_r (ndarray): Absolute position of deputy in inertial frame [m]
            N_rDot (ndarray): Absolute velocity of deputy in inertial frame [m/s]
            time (float): Time corresponding to the deputy's state [s]
        Returns: Hill frame position and velocity of the deputy at this time [m], [m/s]
        """
        N_r_chief, N_v_chief = self.cartesian_state_at_time(time)
        dcm_HN = self.hill_frame_at_time(time)
        N_omega_HN = self.orbit_angular_velocity_at_time(time)

        # Inertial frame relative position and velocity [m], [m/s]
        N_relPosDeputy = N_r - N_r_chief
        N_relVelDeputy = N_rDot - (np.cross(N_omega_HN, N_relPosDeputy) + N_v_chief)

        # Convert to Hill frame
        H_relPosDeputyuty = dcm_HN @ N_relPosDeputy
        H_relVelDeputyuty = dcm_HN @ N_relVelDeputy
        return H_relPosDeputyuty, H_relVelDeputyuty

    @classmethod
    def from_cartesian_state(cls, 
                           position: np.ndarray, 
                           velocity: np.ndarray, 
                           time: float = 0.0,
                           centralBody: str = "Earth") -> 'Orbit':
        """
        Create an Orbit object from Cartesian state.
        
        Args:
            position (np.ndarray): Position vector in inertial frame [m]
            velocity (np.ndarray): Velocity vector in inertial frame [m/s]
            time (float): Time corresponding to the state [s]
            centralBody (str): Central body (default: "Earth")
            
        Returns:
            Orbit: Orbit object with classical orbital elements
        """
        # Set gravitational parameter
        if centralBody == "Earth":
            mu = constants.MU_EARTH_M
        elif centralBody == "Mars":
            mu = constants.MU_MARS
        else:
            raise ValueError(f"Central body '{centralBody}' not supported")
        
        # Calculate orbital elements using the same algorithm as in KeplerianDynamics
        posMag = np.linalg.norm(position)
        velMag = np.linalg.norm(velocity)
        
        h = np.cross(position, velocity)
        angularMomentumMagnitude = np.linalg.norm(h)
        
        K = np.array([0, 0, 1])
        n = np.cross(K, h)
        nMag = np.linalg.norm(n)
        
        # Eccentricity vector
        e_vec = ((velMag**2 - mu / posMag) * position - np.dot(position, velocity) * velocity) / mu
        ecc = np.linalg.norm(e_vec)
        
        # Specific energy
        eta = velMag**2 / 2 - mu / posMag
        
        # Semimajor axis
        if ecc != 1.0:
            sma = -mu / (2 * eta)
        else:
            sma = 0  # Parabolic orbit
        
        # Inclination
        cos_inc = np.clip(h[2] / angularMomentumMagnitude, -1.0, 1.0)
        inc = np.arccos(cos_inc)

        # RAAN
        if nMag < 1e-12:
            # Near equatorial orbit
            raan = 0.0
        else:
            cos_raan = np.clip(n[0] / nMag, -1.0, 1.0)
            raan = np.arccos(cos_raan)
            if n[1] < 0:
                raan = 2 * np.pi - raan
        
        # Argument of periapsis
        if nMag < 1e-12 or ecc < 1e-12:
            # Near equatorial or circular orbit
            argp = 0.0
        else:
            cos_argp = np.clip(np.dot(n, e_vec) / (nMag * ecc), -1.0, 1.0)
            argp = np.arccos(cos_argp)
            if e_vec[2] < 0:
                argp = 2 * np.pi - argp

        # True anomaly
        if ecc < 1e-12:
            # Circular orbit
            nu = 0.0
        else:
            cos_nu = np.clip(np.dot(e_vec, position) / (ecc * posMag), -1.0, 1.0)
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
            
            # Mean anomaly
            M = E - (ecc * np.sin(E))
        else:
            # For parabolic orbits, use a different approach
            M = 0  # Placeholder - would need special handling for parabolic orbits
        
        return cls(semimajorAxis=sma,
                  eccentricity=ecc,
                  inclination=inc,
                  raan=raan,
                  argumentOfPeriapsis=argp,
                  meanAnomalyAtEpoch=M,
                  timeOfEpoch=time,
                  centralBody=centralBody)
    

    @classmethod
    def from_chief_and_delta_oe(cls, 
                                chiefOrbit: 'Orbit',
                                deltaSemimajorAxis: float,
                                deltaEccentricity: float,
                                deltaInclination: float,
                                deltaRaan: float,
                                deltaArgumentOfPeriapsis: float,
                                deltaMeanAnomaly: float) -> 'Orbit':
        """
        Create an Orbit object from the chief orbit and the delta orbit element differences
        """
        sma = chiefOrbit.semimajorAxis + deltaSemimajorAxis
        ecc = chiefOrbit.eccentricity + deltaEccentricity
        inc = chiefOrbit.inclination + deltaInclination
        raan = chiefOrbit.raan + deltaRaan
        argp = chiefOrbit.argumentOfPeriapsis + deltaArgumentOfPeriapsis
        M = chiefOrbit.meanAnomalyAtEpoch + deltaMeanAnomaly
        time = chiefOrbit.timeOfEpoch
        centralBody = chiefOrbit.centralBody

        return cls(semimajorAxis=sma,
                  eccentricity=ecc,
                  inclination=inc,
                  raan=raan,
                  argumentOfPeriapsis=argp,
                  meanAnomalyAtEpoch=M,
                  timeOfEpoch=time,
                  centralBody=centralBody)

    @classmethod
    def delta_oe_from_chief_and_deputy(cls,
                                        chiefOrbit: 'Orbit',
                                        deputyOrbit: 'Orbit') -> tuple[float, float, float, float, float, float]:
        """
        Given an orbit definition for a chief and a deputy, compute the orbit element difference
        description of the deputy relative to the chief 
        NOTE: OE differences here are assumed dOE = deputyOE - chiefOE

        Args:
            chiefOrbit (Orbit): The chief orbit 
            deputyOrbit (Orbit): The deputy orbit
        Returns:
            Tuple of orbit element differences (in m, rad) of form 
            delta_sma, delta_ecc, delta_inc, deltaRaan, deltaArgumentOfPeriapsis, delta_meanAnom 
        """
        deltaSma = deputyOrbit.semimajorAxis - chiefOrbit.semimajorAxis
        deltaEcc = deputyOrbit.eccentricity - chiefOrbit.eccentricity
        deltaInc = deputyOrbit.inclination - chiefOrbit.inclination
        deltaRaan = deputyOrbit.raan - chiefOrbit.raan
        deltaAp = deputyOrbit.argumentOfPeriapsis - chiefOrbit.argumentOfPeriapsis
        deltaMeanAnom = deputyOrbit.meanAnomalyAtEpoch - chiefOrbit.meanAnomalyAtEpoch

        return (deltaSma, deltaEcc, deltaInc, deltaRaan, deltaAp, deltaMeanAnom)

    def __str__(self) -> str:
        """String representation of the orbit."""
        return (f"Orbit(centralBody={self.centralBody}, "
                f"a={self.semimajorAxis:.2e} m, "
                f"e={self.eccentricity:.6f}, "
                f"i={np.degrees(self.inclination):.2f}°, "
                f"Ω={np.degrees(self.raan):.2f}°, "
                f"ω={np.degrees(self.argumentOfPeriapsis):.2f}°, "
                f"M₀={np.degrees(self.meanAnomalyAtEpoch):.2f}°, "
                f"t₀={self.timeOfEpoch:.1f} s)")
    
    def __repr__(self) -> str:
        """Detailed string representation of the orbit."""
        return (f"Orbit(\n"
                f"  semimajorAxis={self.semimajorAxis},\n"
                f"  eccentricity={self.eccentricity},\n"
                f"  inclination={self.inclination},\n"
                f"  raan={self.raan},\n"
                f"  argumentOfPeriapsis={self.argumentOfPeriapsis},\n"
                f"  meanAnomalyAtEpoch={self.meanAnomalyAtEpoch},\n"
                f"  timeOfEpoch={self.timeOfEpoch},\n"
                f"  centralBody='{self.centralBody}'\n"
                f")")