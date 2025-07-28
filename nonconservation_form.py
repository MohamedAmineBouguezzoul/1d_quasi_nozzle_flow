import numpy as np
import matplotlib.pyplot as plt


def initialize_parameters(grid_size=31, domain_length=3):
    """Initialize the computational grid and arrays for flow variables."""
    grid = np.linspace(0, domain_length, grid_size)
    rho = np.zeros(grid_size)  # Density
    V = np.zeros(grid_size)    # Velocity
    T = np.zeros(grid_size)    # Temperature
    A = np.zeros(grid_size)    # Area
    P = np.zeros(grid_size)    # Pressure
    m_dot = np.zeros(grid_size)  # Mass flow rate
    M = np.zeros(grid_size)    # Mach number
    
    # Arrays to collect values at throat (index 15) at different timesteps
    throat_data = {
        'M': np.zeros(0),
        'rho': np.zeros(0),
        'V': np.zeros(0),
        'T': np.zeros(0),
        'P': np.zeros(0)
    }
    
    return grid, rho, V, T, A, P, m_dot, M, throat_data


def set_initial_conditions(grid, rho, V, T, A):
    """Set initial conditions for the flow variables."""
    for i in range(len(grid)):
        rho[i] = 1 - (0.3146 * grid[i])
        T[i] = 1 - (0.2314 * grid[i])
        V[i] = (0.1 + 1.09 * grid[i]) * T[i] ** 0.5
        A[i] = 1 + 2.2 * (grid[i] - 1.5) ** 2


def calculate_time_step(dx, V, T, gamma=1.4):
    """Calculate the time step based on CFL condition."""
    sound_speed = np.sqrt(T)
    t_all = 0.5 * dx / (sound_speed + V)
    return np.min(t_all)


def apply_boundary_conditions(rho, V, T, rho_bar, V_bar, T_bar):
    """Apply inflow and outflow boundary conditions."""
    # Inflow boundary (left)
    rho[0] = rho_bar[0] = 1
    T[0] = T_bar[0] = 1
    V[0] = V_bar[0] = 2 * V[1] - V[2]
    
    # Outflow boundary (right)
    rho[-1] = rho_bar[-1] = 2 * rho[-2] - rho[-3]
    T[-1] = T_bar[-1] = 2 * T[-2] - T[-3]
    V[-1] = V_bar[-1] = 2 * V[-2] - V[-3]


def calculate_derivatives(i, rho, V, T, A, dx, gamma):
    """Calculate time derivatives of flow variables."""
    # Forward differences for derivatives
    drhodt = -(rho[i] * (V[i+1] - V[i]) / dx) - \
             (rho[i] * V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx) - \
             (V[i] * (rho[i+1] - rho[i]) / dx)
    
    dVdt = -(V[i] * (V[i+1] - V[i]) / dx) - \
           ((T[i+1] - T[i]) / (dx * gamma)) - \
           (T[i] * ((rho[i+1] - rho[i]) / (rho[i] * dx * gamma)))
    
    dTdt = -(V[i] * (T[i+1] - T[i]) / dx) - \
           ((gamma - 1) * T[i] * (((V[i+1] - V[i]) / dx) + \
           (V[i] * (np.log(A[i+1]) - np.log(A[i])) / dx)))
    
    return drhodt, dVdt, dTdt


def calculate_barred_derivatives(i, rho_bar, V_bar, T_bar, A, dx, gamma):
    """Calculate time derivatives using barred variables (backward differences)."""
    drhodt_bar = -(rho_bar[i] * (V_bar[i] - V_bar[i-1]) / dx) - \
                 (rho_bar[i] * V_bar[i] * (np.log(A[i]) - np.log(A[i-1])) / dx) - \
                 (V_bar[i] * (rho_bar[i] - rho_bar[i-1]) / dx)
    
    dVdt_bar = -(V_bar[i] * (V_bar[i] - V_bar[i-1]) / dx) - \
               ((T_bar[i] - T_bar[i-1]) / (dx * gamma)) - \
               (T_bar[i] * ((rho_bar[i] - rho_bar[i-1]) / (rho_bar[i] * dx * gamma)))
    
    dTdt_bar = -(V_bar[i] * (T_bar[i] - T_bar[i-1]) / dx) - \
               ((gamma - 1) * T_bar[i] * (((V_bar[i] - V_bar[i-1]) / dx) + \
               (V_bar[i] * (np.log(A[i]) - np.log(A[i-1])) / dx)))
    
    return drhodt_bar, dVdt_bar, dTdt_bar


def update_variables(i, rho, V, T, drhodt, dVdt, dTdt, 
                    drhodt_bar, dVdt_bar, dTdt_bar, dt):
    """Update flow variables using the predictor-corrector method."""
    rho[i] += 0.5 * (drhodt + drhodt_bar) * dt
    V[i] += 0.5 * (dVdt + dVdt_bar) * dt
    T[i] += 0.5 * (dTdt + dTdt_bar) * dt


def record_throat_data(throat_data, rho, V, T, throat_index=15):
    """Record flow variables at the throat location."""
    throat_data['rho'] = np.append(throat_data['rho'], rho[throat_index])
    throat_data['T'] = np.append(throat_data['T'], T[throat_index])
    throat_data['V'] = np.append(throat_data['V'], V[throat_index])


def calculate_derived_quantities(rho, V, T, A, throat_data):
    """Calculate derived quantities like pressure, Mach number, etc."""
    P = rho * T
    m_dot = rho * A * V
    M = V / np.sqrt(T)
    
    # Calculate throat quantities
    throat_data['P'] = throat_data['rho'] * throat_data['T']
    throat_data['M'] = throat_data['V'] / np.sqrt(throat_data['T'])
    
    return P, m_dot, M


def plot_results(grid, rho, T, M, P, A, throat_data, t_array):
    
    # Create a 2D grid for the nozzle visualization
    y_resolution = 50  # Number of points in vertical direction
    x = np.tile(grid, (y_resolution, 1))
    
    # Create nozzle boundaries (top and bottom)
    nozzle_top = A / (2 * np.max(A))  # Normalized to [-0.5, 0.5]
    nozzle_bottom = -nozzle_top
    
    # Create vertical coordinates (from bottom to top for each x)
    y = np.linspace(nozzle_bottom, nozzle_top, y_resolution)
    
    # Create parameter fields (repeating values vertically)
    M_2d = np.tile(M, (y_resolution, 1))
    rho_2d = np.tile(rho, (y_resolution, 1))
    T_2d = np.tile(T, (y_resolution, 1))
    P_2d = np.tile(P, (y_resolution, 1))

    plt.figure(1, figsize=(15, 10))
    
    # Plot 1: Mach Number Contour
    plt.subplot(2, 2, 1)
    plt.contourf(x, y, M_2d, levels=50, cmap='jet')
    plt.colorbar(label='Mach Number')
    plt.plot(grid, nozzle_top, 'k-', linewidth=1.5)
    plt.plot(grid, nozzle_bottom, 'k-', linewidth=1.5)
    plt.fill_between(grid, nozzle_top, nozzle_bottom, color='k', alpha=0.1)
    plt.title('Mach Number Contour')
    plt.xlabel('Axial Position')
    plt.ylabel('Nozzle Height')
    
    # Plot 2: Density Contour
    plt.subplot(2, 2, 2)
    plt.contourf(x, y, rho_2d, levels=50, cmap='viridis')
    plt.colorbar(label='Density')
    plt.plot(grid, nozzle_top, 'k-', linewidth=1.5)
    plt.plot(grid, nozzle_bottom, 'k-', linewidth=1.5)
    plt.title('Density Contour')
    plt.xlabel('Axial Position')
    plt.ylabel('Nozzle Height')
    
    # Plot 3: Temperature Contour
    plt.subplot(2, 2, 3)
    plt.contourf(x, y, T_2d, levels=50, cmap='hot')
    plt.colorbar(label='Temperature')
    plt.plot(grid, nozzle_top, 'k-', linewidth=1.5)
    plt.plot(grid, nozzle_bottom, 'k-', linewidth=1.5)
    plt.title('Temperature Contour')
    plt.xlabel('Axial Position')
    plt.ylabel('Nozzle Height')
    
    # Plot 4: Pressure Contour
    plt.subplot(2, 2, 4)
    plt.contourf(x, y, P_2d, levels=50, cmap='plasma')
    plt.colorbar(label='Pressure')
    plt.plot(grid, nozzle_top, 'k-', linewidth=1.5)
    plt.plot(grid, nozzle_bottom, 'k-', linewidth=1.5)
    plt.title('Pressure Contour')
    plt.xlabel('Axial Position')
    plt.ylabel('Nozzle Height')
    
    plt.tight_layout()

    plt.figure(2, figsize=(15, 10))
    plt.plot(t_array, throat_data['rho'], label="Density", color="green")
    plt.plot(t_array, throat_data['T'], label="Temperature", color="red")
    plt.plot(t_array, throat_data['M'], label="Mach Number", color="black")
    plt.plot(t_array, throat_data['P'], label="Pressure", color="blue")
    plt.title("Throat Parameters Over Time")
    plt.xlabel("Time")
    plt.grid(True)
    plt.legend()
    plt.show()
    


def run_simulation(grid_size=31, domain_length=3, time_steps=1400, gamma=1.4):
    """Main function to run the nozzle flow simulation."""
    # Initialize all parameters
    grid, rho, V, T, A, P, m_dot, M, throat_data = initialize_parameters(grid_size, domain_length)
    
    # Set initial conditions
    set_initial_conditions(grid, rho, V, T, A)
    
    # Calculate grid spacing and initial time step
    dx = grid[1] - grid[0]
    dt = calculate_time_step(dx, V, T, gamma)
    
    # Time array for plotting
    t_array = np.linspace(0, time_steps, time_steps)
    
    # Initialize barred variables
    rho_bar = np.zeros_like(rho)
    V_bar = np.zeros_like(V)
    T_bar = np.zeros_like(T)
    
    # Main time-stepping loop
    for n in range(time_steps):
        # Apply boundary conditions
        apply_boundary_conditions(rho, V, T, rho_bar, V_bar, T_bar)
        
        # Update interior points
        for i in range(1, len(grid)-1):
            # Predictor step: calculate derivatives and barred values
            drhodt, dVdt, dTdt = calculate_derivatives(i, rho, V, T, A, dx, gamma)
            rho_bar[i] = rho[i] + drhodt * dt
            V_bar[i] = V[i] + dVdt * dt
            T_bar[i] = T[i] + dTdt * dt
            
            # Corrector step: calculate barred derivatives and update variables
            drhodt_bar, dVdt_bar, dTdt_bar = calculate_barred_derivatives(
                i, rho_bar, V_bar, T_bar, A, dx, gamma)
            
            update_variables(i, rho, V, T, drhodt, dVdt, dTdt,
                            drhodt_bar, dVdt_bar, dTdt_bar, dt)
        
        # Record throat data
        record_throat_data(throat_data, rho, V, T)
    
    # Calculate derived quantities
    P, m_dot, M = calculate_derived_quantities(rho, V, T, A, throat_data)
    
    # Print some results
    print("Grid:", grid)
    print("Density:", rho)
    print("Temperature:", T)
    print("Velocity:", V)
    print("Pressure:", P)
    print("Area:", A)
    print("Time step:", dt)
    print("Grid spacing:", dx)
    print("Mass flow rate:", m_dot)
    print("Mach number:", M)
    
    # Plot results
    plot_results(grid, rho, T, M, P, A, throat_data, t_array)


if __name__ == "__main__":
    run_simulation()