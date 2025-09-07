"""
Script for modeling the dynamics of a formation using the orbit element difference formulation of
relative motion
"""
import numpy as np

from spacecraft_dynamics.orbit import Orbit, DeltaOeDynamics
from spacecraft_dynamics.utils import true_to_mean_anomaly


def deputy_hill_frame_position_at_true_anomaly(
                                            chiefOrbit:Orbit,
                                            delta_semimajor_axis: float,
                                            delta_eccentricity: float,
                                            delta_inclination: float,
                                            delta_raan: float,
                                            delta_ap: float,
                                            delta_mean_anomaly: float,
                                            trueAnomaly:float
                                            ) -> np.ndarray:
    """
    Given an orbit definition and a true anomaly of the chief, use the linearized
    delta OE solution to compute the relative position of the deputy in the Hill frame
    See eq 14.140 in Schaub, Junkins

    Returns: Hill frame position of the deputy when the chief has this true anomaly [m]
    """

    deltaA, deltaEcc, deltaInc, deltaRaan, deltaAp, deltaMeanAnom = delta_semimajor_axis, delta_eccentricity, delta_inclination, delta_raan, delta_ap, delta_mean_anomaly

    # Get the necessary chief orbit params
    sma = chiefOrbit.semimajor_axis
    ecc = chiefOrbit.eccentricity
    inc = chiefOrbit.inclination

    q1 = ecc * np.cos(chiefOrbit.argument_of_periapsis)
    q2 = ecc * np.sin(chiefOrbit.argument_of_periapsis)
    eta = np.sqrt(1. - (q1 * q1) - (q2 * q2))
    trueLatitude = chiefOrbit.argument_of_periapsis + trueAnomaly
    r = (sma * eta * eta) / (1. + (ecc * np.cos(trueAnomaly)))

    # x/y/z components of position in Hill frame [m]
    posX = (r / sma * deltaA) + (sma * ecc * np.sin(trueAnomaly) / eta * deltaMeanAnom) - (sma * np.cos(trueAnomaly) * deltaEcc)

    posY = (r / (eta * eta * eta) * ((1. + ecc * np.cos(trueAnomaly))**2) * deltaMeanAnom) + (r * deltaAp) + \
        (r * np.sin(trueAnomaly) / (eta * eta) * (2. + ecc * np.cos(trueAnomaly)) * deltaEcc) + (r * np.cos(inc) * deltaRaan)

    posZ = r * (np.sin(trueLatitude) * deltaInc - (np.cos(trueLatitude) * np.sin(inc) * deltaRaan))

    return np.array([posX, posY, posZ])

def main():

    mean_anomaly_at_epoch = true_to_mean_anomaly(true_anomaly=np.radians(10.0), eccentricity=0.2)

    chiefOrbit = Orbit(semimajor_axis=10000e3, 
                        eccentricity=0.2, 
                        inclination=np.radians(37.0),
                        raan=np.radians(40.0),
                        argument_of_periapsis=np.radians(65.0),
                        mean_anomaly_at_epoch=mean_anomaly_at_epoch,
                        time_of_epoch=0.0
                        )



    trueAnomaly_1 = np.radians(10.0)
    trueAnomaly_2 = trueAnomaly_1 + np.radians(60.0)
    H_rho_1 = deputy_hill_frame_position_at_true_anomaly(chiefOrbit=chiefOrbit, 
                                                        delta_semimajor_axis=0.0,
                                                        delta_eccentricity=0.0001,
                                                        delta_inclination=0.001,
                                                        delta_raan=0.001,
                                                        delta_ap=0.001,
                                                        delta_mean_anomaly=-0.001,
                                                        trueAnomaly=trueAnomaly_1)

    H_rho_2 = deputy_hill_frame_position_at_true_anomaly(chiefOrbit=chiefOrbit, 
                                                        delta_semimajor_axis=0.0,
                                                        delta_eccentricity=0.0001,
                                                        delta_inclination=0.001,
                                                        delta_raan=0.001,
                                                        delta_ap=0.001,
                                                        delta_mean_anomaly=-0.001,
                                                        trueAnomaly=trueAnomaly_2)

    print(f"H_rho_0: [{H_rho_1 * 1e-3}] # km")
    print(f"H_rho_1: [{H_rho_2 * 1e-3}] # km")

if __name__ == "__main__":
    main()