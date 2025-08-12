import numpy as np
from spacecraft_dynamics.utils import constants
from spacecraft_dynamics.orbit import Orbit


def mapping_oe_differences_to_relative_cartesian_state(chiefOrbit:Orbit=None, 
                                            trueAnomaly:float=0.0, 
                                            semiMajorAxis:float=None, 
                                            trueLatitude:float=None,
                                            inclination:float=None,
                                            q1:float=None,
                                            q2:float=None,
                                            raan:float=None,
                                            mu:float=constants.MU_EARTH_KM,
                                            units:str='km') -> np.ndarray:
    """
    Given an orbit definition and current true anomaly, generate the linearized mapping
    matrix [A] that maps orbit elemetn differences to relative state coordinates

    deltaX = [A] * deltaOe

    NOTE: The resulting matrix assumes deltaX = [x,y,z,xdot,ydot,zdot]^T and 
    deltaOe = [da, dTheta, di, dq1, dq2, dOmega]^T

    Args:
        chiefOrbit (Orbit): Orbit object for the chief spacecraft, if provided, it is used to 
        populate the other parameters
        trueAnomaly (float): True anomaly of the deputy spacecraft, must be provided if chiefOrbit is provided 
        semiMajorAxis (float): Semi-major axis of the chief spacecraft, must be provided if 
        chiefOrbit is not provided [km or m]
        trueLatitude (float): True latitude of the deputy spacecraft, must be provided 
        if chiefOrbit is not provided [rad]
        inclination (float): Inclination of the chief spacecraft, must be provided 
        if chiefOrbit is not provided [rad]
        q1 (float): q1, must be provided if chiefOrbit is not provided
        q2 (float): q2, must be provided if chiefOrbit is not provided
        raan (float): Right ascension of the ascending node, must be provided if chiefOrbit 
        is not provided [rad]
        mu (float): Standard gravitational parameter of the central body, must be provided 
        if chiefOrbit is not provided [km^3/s^2 or m^3/s^2], defaults to MU_EARTH_KM
        units (str): Units of the input parameters, must be either 'km' or 'm', defaults to 'km'

    Returns:
        np.ndarray: The mapping matrix [A]
    """

    # Exctract orbit params to make expressions more concise
    if chiefOrbit:
        sma = chiefOrbit.semimajor_axis
        ecc = chiefOrbit.eccentricity
        inc = chiefOrbit.inclination
        raan = chiefOrbit.raan
        omega = chiefOrbit.argument_of_periapsis
        trueLat = omega + trueAnomaly
        q1 = ecc * np.cos(omega)
        q2 = ecc * np.sin(omega)
        p = chiefOrbit.semi_latus_rectum
        mu = chiefOrbit.mu

    else:
        # Use the other provided parameters
        sma = semiMajorAxis
        inc = inclination
        raan = raan
        trueLat = trueLatitude
        q1 = q1
        q2 = q2
        ecc = np.sqrt((q1 * q1) + (q2 * q2))
        p = sma * (1.0 - (q1 * q1) - (q2 * q2))
        if units == 'km':
            mu = constants.MU_EARTH_KM
        elif units == 'm':
            mu = constants.MU_EARTH_M

    # Calculate orbit radius
    r = (sma * (1. - (q1 * q1) - (q2 * q2))) / (1. + (q1 * np.cos(trueLat)) + (q2 * np.sin(trueLat)))
    print(f"dOE -> dX:\n r= {r} ecc= {ecc}")
    # Orbit angular momentum
    h = np.sqrt(mu * p)

    # Radial velocity of chief
    vRadial = (h / p) * ((q1 * np.sin(trueLat))- (q2 * np.cos(trueLat)))

    # Transverse velocity of chief
    vTransverse = (h / p) * (1. + (q1 * np.cos(trueLat)) + (q2 * np.sin(trueLat)))

    A = np.zeros((6,6))
    
    # Row for expression of x
    A[0,0] = r / sma
    A[0,1] = vRadial / vTransverse * r
    A[0,2] = 0.
    A[0,3] = -(r / p) * (2. * sma * q1 + r * np.cos(trueLat))
    A[0,4] = -(r / p) * (2. * sma * q2 + r * np.sin(trueLat))
    A[0,5] = 0.

    # Row for expression of y
    A[1,0] = 0.
    A[1,1] = r
    A[1,2] = 0.
    A[1,3] = 0.
    A[1,4] = 0.
    A[1,5] = r * np.cos(inc)

    # Row for z
    A[2,0] = 0.
    A[2,1] = 0.
    A[2,2] = r * np.sin(trueLat)
    A[2,3] = 0.
    A[2,4] = 0.
    A[2,5] = -r * np.cos(trueLat) * np.sin(inc) 

    # Row for xDot
    A[3,0] = -vRadial / (2. * sma)
    A[3,1] = h * ((1. / r) - (1. / p))
    A[3,2] = 0.
    A[3,3] = 1. / p * (vRadial * sma * q1 + h * np.sin(trueLat))
    A[3,4] = 1. / p * (vRadial * sma * q2 - h * np.cos(trueLat))
    A[3,5] = 0.

    # Row for yDot
    A[4,0] = -3. * vTransverse / (2. * sma)
    A[4,1] = -vRadial
    A[4,2] = 0.
    A[4,3] = (1. / p) * (3. * vTransverse * sma * q1 + 2. * h * np.cos(trueLat))
    A[4,4] = (1. / p) * (3. * vTransverse * sma * q2 + 2. * h * np.sin(trueLat))
    A[4,5] = vRadial * np.cos(inc)

    # Row for zDot
    A[5,0] = 0.
    A[5,1] = 0.
    A[5,2] = vTransverse * np.cos(trueLat) + vRadial * np.sin(trueLat)
    A[5,3] = 0.
    A[5,4] = 0.
    A[5,5] = np.sin(inc) * (vTransverse * np.sin(trueLat) - vRadial * np.cos(trueLat))

    return A


def mapping_cartesian_state_to_oe_differences(chiefOrbit:Orbit=None, 
                                            trueAnomaly:float=0.0, 
                                            semiMajorAxis:float=None, 
                                            trueLatitude:float=None,
                                            inclination:float=None,
                                            q1:float=None,
                                            q2:float=None,
                                            raan:float=None,
                                            mu:float=constants.MU_EARTH_KM,
                                            units:str='km') -> np.ndarray:
    """
    Given an orbit definition and current true anomaly, generate the linearized mapping
    matrix [A] that maps orbit elemetn differences to relative state coordinates

    deltaX = [A] * deltaOe

    NOTE: The resulting matrix assumes deltaX = [x,y,z,xdot,ydot,zdot]^T and 
    deltaOe = [da, dTheta, di, dq1, dq2, dOmega]^T

    Args:
        chiefOrbit (Orbit): Orbit object for the chief spacecraft, if provided, it is used to 
        populate the other parameters
        trueAnomaly (float): True anomaly of the deputy spacecraft, must be provided if chiefOrbit is provided 
        semiMajorAxis (float): Semi-major axis of the chief spacecraft, must be provided if 
        chiefOrbit is not provided [km or m]
        trueLatitude (float): True latitude of the deputy spacecraft, must be provided 
        if chiefOrbit is not provided [rad]
        inclination (float): Inclination of the chief spacecraft, must be provided 
        if chiefOrbit is not provided [rad]
        q1 (float): q1, must be provided if chiefOrbit is not provided
        q2 (float): q2, must be provided if chiefOrbit is not provided
        raan (float): Right ascension of the ascending node, must be provided if chiefOrbit 
        is not provided [rad]
        mu (float): Standard gravitational parameter of the central body, must be provided 
        if chiefOrbit is not provided [km^3/s^2 or m^3/s^2], defaults to MU_EARTH_KM
        units (str): Units of the input parameters, must be either 'km' or 'm', defaults to 'km'

    Returns:
        np.ndarray: The mapping matrix [A_inv]
    """

    # Exctract orbit params to make expressions more concise
    if chiefOrbit:
        sma = chiefOrbit.semimajor_axis
        ecc = chiefOrbit.eccentricity
        inc = chiefOrbit.inclination
        raan = chiefOrbit.raan
        omega = chiefOrbit.argument_of_periapsis
        trueLat = omega + trueAnomaly
        q1 = ecc * np.cos(omega)
        q2 = ecc * np.sin(omega)
        p = chiefOrbit.semi_latus_rectum
        mu = chiefOrbit.mu

    else:
        # Use the other provided parameters
        sma = semiMajorAxis
        inc = inclination
        raan = raan
        trueLat = trueLatitude
        q1 = q1
        q2 = q2
        ecc = np.sqrt((q1 * q1) + (q2 * q2))
        p = sma * (1.0 - (q1 * q1) - (q2 * q2))
        if units == 'km':
            mu = constants.MU_EARTH_KM
        elif units == 'm':
            mu = constants.MU_EARTH_M

    # Calculate orbit radius
    r = (sma * (1. - (q1 * q1) - (q2 * q2))) / (1. + (q1 * np.cos(trueLat)) + (q2 * np.sin(trueLat)))
    print(f"dX -> dOE:\n r= {r} ecc= {ecc}")
    # Orbit angular momentum
    h = np.sqrt(mu * p)

    # Radial velocity of chief
    vRadial = (h / p) * ((q1 * np.sin(trueLat))- (q2 * np.cos(trueLat)))

    # Transverse velocity of chief
    vTransverse = (h / p) * (1. + (q1 * np.cos(trueLat)) + (q2 * np.sin(trueLat)))

    # Define nondimensional parameters (see eq G.1 - G.5)
    alpha = sma / r
    v = vRadial / vTransverse
    rho = r / p
    k1 = alpha * ((1. / rho) - 1.)
    k2 = alpha * v * v / rho
    cotInc = 1. / np.tan(inc)
    aInv = np.zeros((6,6))

    # d
    aInv[0,0] = 2. * alpha * (2. + 3. * k1 + 2. * k2)
    aInv[0,1] = -2. * alpha * v * (1. + 2. * k1 + k2)
    aInv[0,2] = 0.
    aInv[0,3] = 2. * alpha * alpha * v * p / vTransverse
    aInv[0,4] = 2. * (sma / vTransverse) * (1. + 2. * k1 + k2)
    aInv[0,5] = 0.

    aInv[1,0] = 0.0
    aInv[1,1] = 1. / r
    aInv[1,2] = ((cotInc) / r) * (np.cos(trueLat) + v * np.sin(trueLat))
    aInv[1,3] = 0.
    aInv[1,4] = 0.
    aInv[1,5] = -np.sin(trueLat) * (cotInc) / vTransverse

    aInv[2,0] = 0.0
    aInv[2,1] = 0.0
    aInv[2,2] = (np.sin(trueLat) - (v * np.cos(trueLat))) / r
    aInv[2,3] = 0.
    aInv[2,4] = 0.
    aInv[2,5] = np.cos(trueLat) / vTransverse

    aInv[3,0] = 1. / (rho * r) * (3. * np.cos(trueLat) + 2. * v * np.sin(trueLat))
    aInv[3,1] = -1. / r * ((v * v * np.sin(trueLat) / rho) + q1 * np.sin(2. * trueLat) - q2 * np.cos(2. * trueLat))
    aInv[3,2] = -q2 * (cotInc) / r * (np.cos(trueLat) + v * np.sin(trueLat))
    aInv[3,3] = np.sin(trueLat) / (rho * vTransverse)
    aInv[3,4] = 1. / (rho * vTransverse) * (2. * np.cos(trueLat) + v * np.sin(trueLat))
    aInv[3,5] = q2 * (cotInc) * np.sin(trueLat) / vTransverse

    aInv[4,0] = 1. / (rho * r) * (3. * np.sin(trueLat) - 2. * v * np.cos(trueLat))
    aInv[4,1] = 1. / r * ((v * v * np.cos(trueLat)) / rho + q2 * np.sin(2. * trueLat) + q1 * np.cos(2. * trueLat))
    aInv[4,2] = q1 * (cotInc) / r * (np.cos(trueLat) + v * np.sin(trueLat))
    aInv[4,3] = -np.cos(trueLat) / (rho * vTransverse)
    aInv[4,4] = 1. / (rho * vTransverse) * (2. * np.sin(trueLat) - v * np.cos(trueLat))
    aInv[4,5] = -q1 * (cotInc) * np.sin(trueLat) / vTransverse

    aInv[5,0] = 0.0
    aInv[5,1] = 0.0
    aInv[5,2] = -(np.cos(trueLat) + v * np.sin(trueLat)) / (r * np.sin(inc))
    aInv[5,3] = 0.
    aInv[5,4] = 0.
    aInv[5,5] = np.sin(trueLat) / (vTransverse * np.sin(inc))

    return aInv