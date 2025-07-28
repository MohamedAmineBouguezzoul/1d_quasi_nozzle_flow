import numpy as np
import matplotlib.pyplot as plt


def initialize_grid(x_min=0, x_max=3, n_points=61):
    """Initialize the spatial grid and area distribution"""
    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]
    A = 1 + 2.2 * (x - 1.5) ** 2
    return x, dx, A


def set_initial_conditions(x, A):
    """Set initial conditions for density, temperature, and velocity"""
    rho = np.zeros_like(x)
    T = np.zeros_like(x)
    
    for i in range(len(x)):
        if x[i] <= 0.5:
            rho[i] = 1.0
            T[i] = 1.0
        elif x[i] <= 1.5:
            rho[i] = 1.0 - 0.366 * (x[i] - 0.5)
            T[i] = 1.0 - 0.167 * (x[i] - 0.5)
        elif x[i] <= 2.1:
            rho[i] = 0.634 - 0.702 * (x[i] - 1.5)
            T[i] = 0.833 - 0.4908 * (x[i] - 1.5)
        else:
            rho[i] = 0.5892 + 0.10228 * (x[i] - 2.1)
            T[i] = 0.93968 + 0.0622 * (x[i] - 2.1)
    
    V = 0.59 / (rho * A)
    return rho, T, V


def conservative_to_primitive(U1, U2, U3, A, gamma):
    """Convert conservative variables to primitive variables"""
    rho = U1 / A
    V = U2 / U1
    T = (gamma - 1) * (U3 / U1 - 0.5 * gamma * V**2)
    return rho, V, T


def primitive_to_conservative(rho, V, T, A, gamma):
    """Convert primitive variables to conservative variables"""
    U1 = rho * A
    U2 = rho * V * A
    U3 = rho * A * (T / (gamma - 1) + 0.5 * gamma * V**2)
    return U1, U2, U3


def compute_fluxes(U1, U2, U3, gamma):
    """Compute flux vectors F1, F2, F3"""
    F1 = U2
    F2 = U2**2 / U1 + (gamma - 1)/gamma * (U3 - 0.5 * gamma * U2**2 / U1)
    F3 = gamma * U2 * U3 / U1 - gamma * (gamma - 1) * U2**3 / (2 * U1**2)
    return F1, F2, F3


def compute_geometry_source_term(rho, T, A, dx, gamma):
    """Compute geometry source term J2"""
    dAdx = np.zeros_like(A)
    dAdx[:-1] = (A[1:] - A[:-1]) / dx
    J2 = rho * T * dAdx / gamma
    return J2


def compute_artificial_viscosity(U, P, Cx, dx):
    """Compute artificial viscosity terms"""
    S = np.zeros_like(U)
    for i in range(1, len(U)-1):
        S[i] = Cx * (U[i+1] - 2*U[i] + U[i-1]) * np.abs(P[i+1] - 2*P[i] + P[i-1]) / (P[i+1] + 2*P[i] + P[i-1])
    return S


def apply_boundary_conditions(U1, U2, U3, A, V, T, gamma, P_e):
    """Apply boundary conditions"""
    # Inlet boundary conditions
    U1[0] = A[0]
    U2[0] = 2 * U2[1] - U2[2]
    U3[0] = U1[0] * (T[0] / (gamma - 1) + 0.5 * gamma * V[0] ** 2)
    
    # Outlet boundary conditions
    U1[-1] = 2 * U1[-2] - U1[-3]
    U2[-1] = 2 * U2[-2] - U2[-3]
    U3[-1] = (P_e * A[-1] / (gamma - 1)) + (gamma * U2[-1] * V[-1] / 2)
    
    return U1, U2, U3


def maccormack_step(U1, U2, U3, A, dx, dt, gamma, Cx, P_e):
    """Perform a single MacCormack predictor-corrector step"""
    # Compute primitive variables
    rho, V, T = conservative_to_primitive(U1, U2, U3, A, gamma)
    P = rho * T
    
    # Compute fluxes and source terms
    F1, F2, F3 = compute_fluxes(U1, U2, U3, gamma)
    J2 = compute_geometry_source_term(rho, T, A, dx, gamma)
    
    # Compute artificial viscosity
    S1 = compute_artificial_viscosity(U1, P, Cx, dx)
    S2 = compute_artificial_viscosity(U2, P, Cx, dx)
    S3 = compute_artificial_viscosity(U3, P, Cx, dx)
    
    # Predictor step
    U1_bar = U1 - dt * np.diff(F1, prepend=F1[0]) / dx + S1
    U2_bar = U2 - dt * (np.diff(F2, prepend=F2[0]) / dx - J2) + S2
    U3_bar = U3 - dt * np.diff(F3, prepend=F3[0]) / dx + S3
    
    # Compute primitive variables for predicted state
    rho_bar, V_bar, T_bar = conservative_to_primitive(U1_bar, U2_bar, U3_bar, A, gamma)
    P_bar = rho_bar * T_bar
    
    # Compute fluxes and source terms for predicted state
    F1_bar, F2_bar, F3_bar = compute_fluxes(U1_bar, U2_bar, U3_bar, gamma)
    J2_bar = compute_geometry_source_term(rho_bar, T_bar, A, dx, gamma)
    
    # Compute artificial viscosity for predicted state
    S1_bar = compute_artificial_viscosity(U1_bar, P_bar, Cx, dx)
    S2_bar = compute_artificial_viscosity(U2_bar, P_bar, Cx, dx)
    S3_bar = compute_artificial_viscosity(U3_bar, P_bar, Cx, dx)
    
    # Corrector step
    U1 = 0.5 * (U1 + U1_bar - dt * np.diff(F1_bar, append=F1_bar[-1]) / dx) + S1_bar
    U2 = 0.5 * (U2 + U2_bar - dt * (np.diff(F2_bar, append=F2_bar[-1]) / dx - J2_bar)) + S2_bar
    U3 = 0.5 * (U3 + U3_bar - dt * np.diff(F3_bar, append=F3_bar[-1]) / dx) + S3_bar
    
    # Apply boundary conditions
    U1, U2, U3 = apply_boundary_conditions(U1, U2, U3, A, V, T, gamma, P_e)
    
    return U1, U2, U3


def simulate_nozzle_flow(nt=1600, CFL=0.5, gamma=1.4, Cx=0.2, P_e=0.6784):
    """Main function to run the nozzle flow simulation"""
    # Initialize grid and area distribution
    x, dx, A = initialize_grid()
    
    # Set initial conditions
    rho, T, V = set_initial_conditions(x, A)
    
    # Convert to conservative variables
    U1, U2, U3 = primitive_to_conservative(rho, V, T, A, gamma)
    
    # Time integration
    for t in range(nt):
        # Compute time step
        a = np.sqrt(T)
        dt = CFL * dx / np.max(a + V)
        
        # Perform MacCormack step
        U1, U2, U3 = maccormack_step(U1, U2, U3, A, dx, dt, gamma, Cx, P_e)
        
        # Update primitive variables for next iteration
        rho, V, T = conservative_to_primitive(U1, U2, U3, A, gamma)
    
    # Post-processing
    P = rho * T
    M = V / np.sqrt(T)
    m_dot = rho * V * A
    
    return x, A, rho, V, T, P, M, m_dot, U1, U2, U3, dt

def plot_results(x, A, rho, V, T, P, M, m_dot):
    """Plot the results of the nozzle flow simulation"""
    
    # Create 2D nozzle visualization data
    y_resolution = 50
    x_2d = np.tile(x, (y_resolution, 1))
    nozzle_top = A / (2 * np.max(A))
    nozzle_bottom = -nozzle_top
    y = np.array([np.linspace(b, t, y_resolution) for b, t in zip(nozzle_bottom, nozzle_top)]).T
    
    # Create 2D parameter fields
    M_2d = np.tile(M, (y_resolution, 1))
    P_2d = np.tile(P, (y_resolution, 1))
    rho_2d = np.tile(rho, (y_resolution, 1))
    T_2d = np.tile(T, (y_resolution, 1))
    
    # Figure 1: Contour plots
    fig1 = plt.figure(1, figsize=(15, 10))
    fig1.suptitle('Nozzle Flow Contour Plots', fontsize=16)
    
    # Mach Number Contour
    ax1 = fig1.add_subplot(2, 2, 1)
    cont1 = ax1.contourf(x_2d, y, M_2d, levels=50, cmap='jet')
    fig1.colorbar(cont1, ax=ax1, label='Mach Number')
    ax1.plot(x, nozzle_top, 'k-', linewidth=1.5)
    ax1.plot(x, nozzle_bottom, 'k-', linewidth=1.5)
    ax1.fill_between(x, nozzle_top, nozzle_bottom, color='k', alpha=0.1)
    ax1.set_title('Mach Number Contour')
    ax1.set_xlabel('Axial Position')
    ax1.set_ylabel('Normalized Nozzle Height')
    
    # Pressure Contour
    ax2 = fig1.add_subplot(2, 2, 2)
    cont2 = ax2.contourf(x_2d, y, P_2d, levels=50, cmap='viridis')
    fig1.colorbar(cont2, ax=ax2, label='Pressure')
    ax2.plot(x, nozzle_top, 'k-', linewidth=1.5)
    ax2.plot(x, nozzle_bottom, 'k-', linewidth=1.5)
    ax2.set_title('Pressure Contour')
    ax2.set_xlabel('Axial Position')
    
    # Density Contour
    ax3 = fig1.add_subplot(2, 2, 3)
    cont3 = ax3.contourf(x_2d, y, rho_2d, levels=50, cmap='plasma')
    fig1.colorbar(cont3, ax=ax3, label='Density')
    ax3.plot(x, nozzle_top, 'k-', linewidth=1.5)
    ax3.plot(x, nozzle_bottom, 'k-', linewidth=1.5)
    ax3.set_title('Density Contour')
    ax3.set_xlabel('Axial Position')
    
    # Temperature Contour
    ax4 = fig1.add_subplot(2, 2, 4)
    cont4 = ax4.contourf(x_2d, y, T_2d, levels=50, cmap='inferno')
    fig1.colorbar(cont4, ax=ax4, label='Temperature')
    ax4.plot(x, nozzle_top, 'k-', linewidth=1.5)
    ax4.plot(x, nozzle_bottom, 'k-', linewidth=1.5)
    ax4.set_title('Temperature Contour')
    ax4.set_xlabel('Axial Position')
    
    plt.tight_layout()
    
    # Figure 2: Line plots
    fig2 = plt.figure(2, figsize=(12, 8))
    fig2.suptitle('Nozzle Flow Parameters', fontsize=16)
    
    # Create subplots for line plots
    ax5 = fig2.add_subplot(2, 2, 1)
    ax6 = fig2.add_subplot(2, 2, 2)
    ax7 = fig2.add_subplot(2, 2, 3)
    ax8 = fig2.add_subplot(2, 2, 4)
    
    # Mach number
    ax5.plot(x, M, 'r-', linewidth=2)
    ax5.set_title('Mach Number')
    ax5.set_xlabel('Axial Position')
    ax5.grid(True)
    
    # Pressure
    ax6.plot(x, P, 'b-', linewidth=2)
    ax6.set_title('Pressure')
    ax6.set_xlabel('Axial Position')
    ax6.grid(True)
    
    # Density
    ax7.plot(x, rho, 'g-', linewidth=2)
    ax7.set_title('Density')
    ax7.set_xlabel('Axial Position')
    ax7.grid(True)
    
    # Temperature
    ax8.plot(x, T, 'm-', linewidth=2)
    ax8.set_title('Temperature')
    ax8.set_xlabel('Axial Position')
    ax8.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run the simulation and plot results"""
    # Run simulation
    x, A, rho, V, T, P, M, m_dot, U1, U2, U3, dt = simulate_nozzle_flow()
    
    # Print some results
    print("Area:", A)
    print("Density:", rho)
    print("Velocity:", V)
    print("Temperature:", T)
    print("Pressure:", P)
    print("Mach number:", M)
    print("Mass flow rate:", m_dot)
    print("U1:", U1)
    print("U2:", U2)
    print("U3:", U3)
    print("Final time step:", dt)
    
    # Plot results
    plot_results(x, A, rho, V, T, P, M, m_dot)


if __name__ == "__main__":
    main()