import numpy as np

if __name__ == "__main__":

    """
    CASE 1:
    Given rectilinear position of chief and relative position of deputy, map to
    curvilinear coordinates
    """

    # Position and velocity of chief [km], [km/s]
    r = 7000
    r_dot = 0.05

    # Relative position and velocity of deputy [km], [km/s]
    x = 10
    y = 500
    x_dot = 0.1
    y_dot = -0.1

    # Map to cylindrical coords (without small angle approx)
    delta_r = np.sqrt((r + x)**2 + y**2) - r
    s = r * np.arctan2(y, r + x)
    
    delta_r_dot = ((r + x)*(r_dot + x_dot) + (y * y_dot)) / (delta_r + r) - r_dot
    s_dot = r_dot * np.arctan2(y, r + x) + r * (((r + x) * y_dot) - y * (r_dot + x_dot)) / ((r + x)**2 + y**2)

    print(f"CASE 1:")
    print(f"delta_r = [{delta_r}]")
    print(f"delta_r_dot = [{delta_r_dot}]")
    print(f"s = [{s}]")
    print(f"s_dot = [{s_dot}]")

    """
    CASE 1:
    Given curvilinear position of chief and relative position of deputy, map to
    rectilinear coordinates
    """

    # Position and velocity of chief [km], [km/s]
    r = 7000
    r_dot = 0.05

    # Curvilinear position and velocity of deputy [km], [km/s]
    delta_r = 10
    s = 500
    delta_r_dot = 0.1
    s_dot = -0.1

    # Map to cylindrical coords (without small angle approx)
    x = (r + delta_r) * np.cos(s / r) - r
    # y = (r + delta_r) * np.tan(s / r) # correct
    y = (r + delta_r) * np.sin(s / r)

    x_dot = (delta_r_dot + r_dot) * np.cos(s / r) - ((r + delta_r) * np.sin(s / r) * (s_dot * r - s * r_dot) / (r**2)) - r_dot
    y_dot = (r_dot + delta_r_dot) * np.sin(s / r) + ((r + delta_r)*np.cos(s / r )* (s_dot * r - s * r_dot) / (r**2))

    print(f"CASE 2:")
    print(f"x = [{x}]")
    print(f"x_dot = [{x_dot}]")
    print(f"y = [{y}]")
    print(f"y_dot = [{y_dot}]")