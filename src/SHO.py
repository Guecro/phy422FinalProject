import numpy as np

def SHO(x, v, omega):
    dx_dt = v
    dv_dt = -omega ** 2 * x
    return dx_dt, dv_dt

# Explicit Euler - non-symplectic
def solve_euler(
    x0: float,
    v0: float,
    omega: float,
    t_start: float,
    t_stop: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the SHO with the forward Euler method.

    Parameters
    ----------
    x0:
        Initial position value x(t_start).
    v0:
        Initial velocity value v(t_start).
    omega:
        Angular frequency of the motion.
    t_start:
        Start of the integration interval.
    t_stop:
        End of the integration interval.
    num_steps:
        Number of uniform Euler steps.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(t_points, x_points, v_points)`` containing the stored solution values.

    Notes
    -----
    This is the forward Euler method, which has local truncation error of order
    h^2 and cumulative global error of order h.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if t_stop <= t_start:
        raise ValueError("t_stop must be greater than t_start")

    step_size = (t_stop - t_start) / num_steps
    t_points = np.linspace(t_start, t_stop, num_steps, endpoint=False)
    x_points = np.empty(num_steps, dtype=float)
    v_points = np.empty(num_steps, dtype=float)

    x_value = float(x0)
    v_value = float(v0)
    
    x_points[0] = x_value
    v_points[0] = v_value
    
    for i in range(1, num_steps):
        dx_dt, dv_dt = SHO(x_value, v_value, omega)
        
        x_value += step_size * dx_dt
        v_value += step_size * dv_dt
        
        x_points[i] = x_value
        v_points[i] = v_value
        
    return t_points, x_points, v_points

# RK4 - non-symplectic
def solve_rk4():
    pass

# Symplectic Euler
def solve_euler_symplectic():
    pass

# Velocity Verlet / leapfrog
def solve_verlet():
    pass
