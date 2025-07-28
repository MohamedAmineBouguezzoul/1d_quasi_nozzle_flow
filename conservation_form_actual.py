import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(0, 3, 31)
dx = x[1] - x[0]
A = 1 + 2.2 * (x - 1.5) ** 2

# Constants
gamma = 1.4
CFL = 0.5
nt = 1400

# Initial conditions
rho = np.zeros_like(x)
T = np.zeros_like(x)
V = np.zeros_like(x)

for i in range(len(x)):
    if x[i] <= 0.5:
        rho[i] = 1.0
        T[i] = 1.0
    elif x[i] <= 1.5:
        rho[i] = 1.0 - 0.366 * (x[i] - 0.5)
        T[i] = 1.0 - 0.167 * (x[i] - 0.5)
    else:
        rho[i] = 0.634 - 0.3879 * (x[i] - 1.5)
        T[i] = 0.833 - 0.3507 * (x[i] - 1.5)

V = 0.59 / (rho * A)

# Conservative variables
U1 = rho * A
U2 = rho * V * A
U3 = rho * A * (T / (gamma - 1) + 0.5 * gamma * V**2)

for t in range(nt):
    # Time step
    a = np.sqrt(T)
    dt = CFL * dx / np.max(a + V)

    # Flux vectors
    F1 = U2
    F2 = U2**2 / U1 + (gamma - 1)/gamma * (U3 - 0.5 * gamma * U2**2 / U1)
    F3 = gamma * U2 * U3 / U1 - gamma * (gamma - 1) * U2**3 / (2 * U1**2)

    # Geometry source term
    dAdx = np.zeros_like(x)
    dAdx[:-1] = (A[1:] - A[:-1]) / dx
    J2 = rho * T * dAdx / gamma

    # Predictor
    U1_bar = np.copy(U1)
    U2_bar = np.copy(U2)
    U3_bar = np.copy(U3)

    for i in range(len(x) - 1):
        U1_bar[i] = U1[i] - dt * (F1[i+1] - F1[i]) / dx
        U2_bar[i] = U2[i] - dt * ((F2[i+1] - F2[i]) / dx - J2[i])
        U3_bar[i] = U3[i] - dt * (F3[i+1] - F3[i]) / dx

    rho_bar = U1_bar / A
    V_bar = U2_bar / U1_bar
    T_bar = (gamma - 1) * (U3_bar / U1_bar - 0.5 * gamma * V_bar**2)

    # Recompute fluxes
    F1_bar = U2_bar
    F2_bar = U2_bar**2 / U1_bar + (gamma - 1)/gamma * (U3_bar - 0.5 * gamma * U2_bar**2 / U1_bar)
    F3_bar = gamma * U2_bar * U3_bar / U1_bar - gamma * (gamma - 1) * U2_bar**3 / (2 * U1_bar**2)

    dAdx[1:] = (A[1:] - A[:-1]) / dx
    J2_bar = rho_bar * T_bar * dAdx / gamma

    # Corrector
    for i in range(1, len(x)):
        U1[i] = 0.5 * (U1[i] + U1_bar[i] - dt * (F1_bar[i] - F1_bar[i - 1]) / dx)
        U2[i] = 0.5 * (U2[i] + U2_bar[i] - dt * ((F2_bar[i] - F2_bar[i - 1]) / dx - J2_bar[i]))
        U3[i] = 0.5 * (U3[i] + U3_bar[i] - dt * (F3_bar[i] - F3_bar[i - 1]) / dx)

    # Boundary conditions
    U1[0] = A[0]
    U2[0] = 2 * U2[1] - U2[2]
    U3[0] = U1[0] * (T[0] / (gamma - 1) + 0.5 * gamma * V[0] ** 2)

    U1[-1] = 2 * U1[-2] - U1[-3]
    U2[-1] = 2 * U2[-2] - U2[-3]
    U3[-1] = 2 * U3[-2] - U3[-3]

    # Update primitive variables
    rho = U1 / A
    V = U2 / U1
    T = (gamma - 1) * (U3 / U1 - 0.5 * gamma * V**2)

# Post-processing
P = rho * T
M = V / np.sqrt(T)
m_dot = rho * V * A

print(A)
print(rho)
print(V)
print(T)
print(P)
print(M)
print(m_dot)
print(U1)
print(U2)
print(U3)
print(dt)

# Plot
plt.plot(x, M, label="Mach")
plt.plot(x, P, label="Pressure")
plt.plot(x, rho, label="Density")
plt.plot(x, T, label="Temperature")
plt.xlabel("x")
plt.grid()
plt.legend()
plt.title("MacCormack: Quasi-1D Nozzle Flow")
plt.show()
