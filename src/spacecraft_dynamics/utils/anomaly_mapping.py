
import math

def true_to_mean_anomaly(true_anomaly:float, eccentricity:float) -> float:
    """
    Convert true anomaly to mean anomaly for an elliptical orbit.
    
    Parameters:
    true_anomaly (float): True anomaly in radians
    eccentricity (float): Eccentricity of the orbit (0 < e < 1 for elliptical orbits)
    
    Returns:
    float: Mean anomaly in radians
    """
    # Input validation
    if not 0 <= eccentricity < 1:
        raise ValueError("Eccentricity must be between 0 and 1 for elliptical orbits")
    
    # Convert true anomaly to radians
    nu = true_anomaly
    
    # Calculate eccentric anomaly (E) from true anomaly
    cos_nu = math.cos(nu)
    sin_nu = math.sin(nu)
    cos_E = (eccentricity + cos_nu) / (1 + eccentricity * cos_nu)
    sin_E = math.sqrt(1 - eccentricity**2) * sin_nu / (1 + eccentricity * cos_nu)
    
    # Get eccentric anomaly in radians
    E = math.atan2(sin_E, cos_E)
    if E < 0:
        E += 2 * math.pi  # Ensure E is in [0, 2pi]
    
    # Calculate mean anomaly (M) using Kepler's equation: M = E - e * sin(E)
    M = E - eccentricity * math.sin(E)
    if M < 0:
        M += 2 * math.pi  # Ensure M is in [0, 2pi]
    
    return M