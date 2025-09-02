import numpy as np
import matplotlib.pyplot as plt

# Constants
GAMMA = 1.4
P_EXIT = 0.93
THROAT_INDEX = 15  # Index corresponding to x=1.5 in grid

def initialize_arrays(grid_size):
    """Initialize all arrays with zeros"""
    grid = np.linspace(0, 3, grid_size)
    rho = np.zeros_like(grid)
    V = np.zeros_like(grid)
    T = np.zeros_like(grid)
    A = np.zeros_like(grid)
    P = np.zeros_like(grid)
    m_dot = np.zeros_like(grid)
    M = np.zeros_like(grid)
    return grid, rho, V, T, A, P, m_dot, M

def set_initial_conditions(grid, rho, V, T, A):
    """Set initial conditions for the simulation"""
    for i in range(len(rho)):
        rho[i] = 1 - (0.023 * grid[i])
        T[i] = 1 - (0.009333 * grid[i])
        V[i] = 0.05 + (0.11 * grid[i])
        if grid[i] < 1.5:
            A[i] = 1 + 2.2 * (grid[i] - 1.5) ** 2
        else:
            A[i] = 1 + 0.2223 * (grid[i] - 1.5) ** 2
    return rho, V, T, A

def calculate_time_step(dx, T, V):
    """Calculate the time step based on CFL condition"""
    t_all = 0.5 * dx / (np.sqrt(T) + V)
    return np.min(t_all)

def apply_boundary_conditions(rho, V, T, rho_bar, V_bar, T_bar, P_e):
    """Apply boundary conditions at inlet and outlet"""
    # Inflow boundary conditions
    rho[0] = 1
    T[0] = 1
    V[0] = 2 * V[1] - V[2]
    rho_bar[0] = 1
    T_bar[0] = 1
    V_bar[0] = 2 * V[1] - V[2]

    # Outflow boundary conditions
    rho[-1] = 2 * rho[-2] - rho[-3]
    T[-1] = P_e / rho[-1]
    V[-1] = 2 * V[-2] - V[-3]
    rho_bar[-1] = 2 * rho[-2] - rho[-3]
    T_bar[-1] = P_e / rho[-1]
    V_bar[-1] = 2 * V[-2] - V[-3]
    
    return rho, V, T, rho_bar, V_bar, T_bar

def calculate_derivatives(i, rho, V, T, A, dx, gamma):
    """Calculate time derivatives for the governing equations"""
    drhodt = -(rho[i] * (V[i+1] - V[i]) / dx) - (rho[i] * V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx) - (
            V[i] * (rho[i+1] - rho[i]) / dx)
    dVdt = -(V[i] * (V[i+1] - V[i]) / dx) - ((T[i+1] - T[i]) / (dx * gamma)) - (
            T[i] * ((rho[i+1] - rho[i]) / (rho[i] * dx * gamma)))
    dTdt = -(V[i] * (T[i+1] - T[i]) / dx) - (
            (gamma - 1) * T[i] * (((V[i+1] - V[i]) / dx) + (V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx)))
    return drhodt, dVdt, dTdt

def update_variables(i, rho, V, T, rho_bar, V_bar, T_bar, dt, A, dx, gamma):
    """Update variables using MacCormack scheme"""
    # First predictor step
    drhodt, dVdt, dTdt = calculate_derivatives(i, rho, V, T, A, dx, gamma)
    rho_bar[i] = rho[i] + drhodt * dt
    V_bar[i] = V[i] + dVdt * dt
    T_bar[i] = T[i] + dTdt * dt
    
    # Second corrector step
    drhodt_bar = -(rho_bar[i] * (V_bar[i] - V_bar[i-1]) / dx) - (
            rho_bar[i] * V_bar[i] * (np.log(A[i]) - np.log(A[i-1])) / dx) - (
                 V_bar[i] * (rho_bar[i] - rho_bar[i-1]) / dx)
    dVdt_bar = -(V_bar[i] * (V_bar[i] - V_bar[i-1]) / dx) - ((T_bar[i] - T_bar[i-1]) / (dx * gamma)) - (
            T_bar[i] * ((rho_bar[i] - rho_bar[i-1]) / (rho_bar[i] * dx * gamma)))
    dTdt_bar = -(V_bar[i] * (T_bar[i] - T_bar[i-1]) / dx) - ((gamma - 1) * T_bar[i] * (
            ((V_bar[i] - V_bar[i-1]) / dx) + (V_bar[i] * (np.log(A[i]) - np.log(A[i-1])) / dx)))

    # Final update
    rho[i] += 0.5 * (drhodt + drhodt_bar) * dt
    V[i] += 0.5 * (dVdt + dVdt_bar) * dt
    T[i] += 0.5 * (dTdt + dTdt_bar) * dt
    
    return rho, V, T

def calculate_secondary_variables(rho, V, T, A):
    """Calculate secondary variables: Pressure, Mach number, mass flow rate"""
    P = rho * T
    m_dot = rho * A * V
    M = V / np.sqrt(T)
    return P, m_dot, M

def run(P_e=P_EXIT):
    # Grid generation and initialization
    grid_size = 31
    grid, rho, V, T, A, P, m_dot, M = initialize_arrays(grid_size)
    
    # Set initial conditions
    rho, V, T, A = set_initial_conditions(grid, rho, V, T, A)
    
    # Calculate time step
    dx = grid[1] - grid[0]
    dt = calculate_time_step(dx, T, V)
    
    # Initialize arrays for throat history
    rho_throat = np.array([])
    V_throat = np.array([])
    T_throat = np.array([])
    
    # Time stepping
    t_step = 5000
    t_array = np.linspace(0, t_step, 5000)
    
    for n in range(t_step):
        # Initialize barred variables
        rho_bar = np.zeros_like(rho)
        V_bar = np.zeros_like(V)
        T_bar = np.zeros_like(T)
        
        # Apply boundary conditions
        rho, V, T, rho_bar, V_bar, T_bar = apply_boundary_conditions(
            rho, V, T, rho_bar, V_bar, T_bar, P_e)
        
        # Update interior points
        for i in range(1, grid_size-1):
            rho, V, T = update_variables(
                i, rho, V, T, rho_bar, V_bar, T_bar, dt, A, dx, GAMMA)
        
        # Store throat values
        rho_throat = np.append(rho_throat, rho[THROAT_INDEX])
        V_throat = np.append(V_throat, V[THROAT_INDEX])
        T_throat = np.append(T_throat, T[THROAT_INDEX])
    
    # Calculate secondary variables
    P, m_dot, M = calculate_secondary_variables(rho, V, T, A)
    P_throat = rho_throat * T_throat
    M_throat = V_throat / np.sqrt(T_throat)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Throat parameters over time
    plt.subplot(1, 2, 1)
    plt.plot(t_array, rho_throat, label="Density", color="green")
    plt.plot(t_array, T_throat, label="Temperature", color="red")
    plt.plot(t_array, M_throat, label="Mach Number", color="black")
    plt.plot(t_array, P_throat, label="Pressure", color="blue")
    plt.title("Throat Parameters vs Time")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()
    
    # Final steady state distribution
    plt.subplot(1, 2, 2)
    plt.plot(grid, rho, label="Density", color="green")
    plt.plot(grid, T, label="Temperature", color="red")
    plt.plot(grid, M, label="Mach Number", color="black")
    plt.plot(grid, P, label="Pressure", color="blue")
    plt.title("Steady State Distribution")
    plt.xlabel("Position")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()
    
