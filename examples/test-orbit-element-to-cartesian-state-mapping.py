"""
Script to test the mapping of orbit element differences to relative cartesian state
and vice versa
"""

import numpy as np
from spacecraft_dynamics.utils import (
    mapping_oe_differences_to_relative_cartesian_state,
    mapping_cartesian_state_to_oe_differences
)

np.set_printoptions(precision=2)


def main():

    sma = 7500.0
    trueLat = np.deg2rad(13.0)
    inc = np.deg2rad(22.0)
    q1 = 0.00707107
    q2 = 0.00707107
    raan = np.deg2rad(70.0)

    oeToCartesian = mapping_oe_differences_to_relative_cartesian_state(semiMajorAxis=sma,
                                                                        trueLatitude=trueLat,
                                                                        inclination=inc,
                                                                        q1=q1,
                                                                        q2=q2,
                                                                        raan=raan)
    # Print each row of the resulting array
    for i in range(6):
        row_values = oeToCartesian[i, :].tolist()
        formatted_values = [f"{val:.8e}" for val in row_values]
        print(f"row{i+1} = [{', '.join(formatted_values)}]")

    print(f"---------------------------")
    cartesianToOe = mapping_cartesian_state_to_oe_differences(semiMajorAxis=sma,
                                                            trueLatitude=trueLat,
                                                            inclination=inc,
                                                            q1=q1,
                                                            q2=q2,
                                                            raan=raan)
    for i in range(6):
        row_values = cartesianToOe[i, :].tolist()
        formatted_values = [f"{val:.8e}" for val in row_values]
        print(f"row{i+1} = [{', '.join(formatted_values)}]")


    # validate [A] * [A]^-1 is nearly [I]
    print(f"A * Ainv =\n{np.matmul(oeToCartesian, cartesianToOe)}")

if __name__ == "__main__":
    main()