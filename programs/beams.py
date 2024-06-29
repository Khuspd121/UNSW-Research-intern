import matplotlib.pyplot as plt
import numpy as np

# various parameters
w0 = 1
lambda_ = 1
k = 2 * np.pi / lambda_
zR = np.pi * w0 ** 2 / lambda_


def w(w0, z, zR):
    return w0 * (1 + (z / zR) ** 2) ** 0.5


z = 0
# choose a specific value of z
wZ = w(w0, z, zR)

# grid
x0 = np.linspace(-2, 2, 512)
X, Y = np.meshgrid(x0, x0)
r = np.sqrt(X ** 2 + Y ** 2)
theta = np.arctan2(Y, X)

# Gaussian beam profile(l=0,p=0)
LG_0_0 = np.sqrt(2 / np.pi) * (1 / wZ) * np.exp(-r ** 2 / wZ ** 2)

# vortex beam (l=1,p=0)
LG_1_0 = np.sqrt(2 / np.pi) * (1 / wZ) * (np.sqrt(2) * r / wZ) * np.exp(-r ** 2 / wZ ** 2) * np.exp(1j * theta)

# plotting

plt.imshow(np.abs(LG_0_0) ** 2, extent=(-2, 2, -2, 2), cmap="inferno")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title("gaussian beam")
plt.show()

plt.imshow(np.abs(LG_1_0) ** 2, extent=(-2, 2, -2, 2), cmap="inferno")
plt.title("vortex beam")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

plt.imshow(np.angle(LG_1_0), extent=(-np.pi, np.pi, -np.pi, np.pi), cmap="gray")
plt.title("phase of vortex")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
