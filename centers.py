import numpy as np
import matplotlib.pyplot as plt

# Data range and number of centers
x = np.linspace(0, 1, 400)
M = 10
centers = np.linspace(0, 1, M)

# Heuristic for width: spacing between centers
sigma = centers[1] - centers[0]

def gaussian_basis(x, center, sigma):
    return np.exp(-0.5 * ((x - center) / sigma)**2)

plt.figure(figsize=(8, 4))
for c in centers:
    plt.plot(x, gaussian_basis(x, c, sigma), label=f'Center = {c:.2f}')

# Mark the centers on the plot
plt.scatter(centers, np.ones_like(centers)*0.05, color='red', zorder=5, label='Centers')
plt.title("Gaussian Basis Functions with Evenly Spaced Centers")
plt.xlabel("x")
plt.ylabel("Basis Function Value")
plt.legend()
plt.show()
