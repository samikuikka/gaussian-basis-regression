import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(0, 1, 400)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.sin(2 * np.pi * x) + noise

def gaussian_basis(x, centers, width):
    return np.exp(-(x[:, None] - centers)**2 / (2 * width**2))

M = 10
centers = np.linspace(0, 1, M)
sigma = (centers[1] - centers[0])
Phi = gaussian_basis(x, centers, sigma)

def solve_normal_equations(Phi, y):
    return np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y

w = solve_normal_equations(Phi, y)

x_plot = np.linspace(0, 1, 200)
Phi_plot = gaussian_basis(x_plot, centers, sigma)
y_plot = Phi_plot @ w

plt.figure(figsize=(8, 4))
plt.scatter(x, y, color="blue", label="Data", s=20)
plt.plot(x_plot, y_plot, color="red", label="Gaussian Basis Fit", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Basis Regression via Normal Equations")
plt.legend()
plt.show()