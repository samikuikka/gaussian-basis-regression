import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 data points uniformly between 0 and 1.
x = np.linspace(0, 1, 100)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.sin(2 * np.pi * x) + noise

# Put into a DataFrame (optional)
df = pd.DataFrame({'x': x, 'y': y})

# Plot the data
plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="Data", color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic Data")
plt.legend()
plt.show()
