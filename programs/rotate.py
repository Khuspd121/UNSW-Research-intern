import numpy as np
import matplotlib.pyplot as plt

def gaussian_beam_field(x, y, w0):
    """Generate a Gaussian beam field at z = 0"""
    return np.exp(-2 * (x**2 + y**2) / w0**2)

def fresnel_propagate(U, z, lambda_, dx):
    """Fresnel propagation of the field U over a distance z"""
    N = U.shape[0]
    k = 2 * np.pi / lambda_
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)

    # Fresnel transfer function
    H = np.exp(-1j * np.pi * lambda_ * z * (FX ** 2 + FY ** 2))

    # Propagate the field
    U1 = np.fft.ifft2(np.fft.fft2(U) * H)
    return U1

def rotate_coordinates(x, y, angle):
    """Rotate coordinates by the given angle"""
    theta = np.radians(angle)
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

# Define parameters
w0 = 1.0  # Beam waist
lambda_ = 1.0  # Wavelength
z = 1.0  # Propagation distance
dx = 0.05  # Grid spacing
N = 400  # Grid size
angle = 45  # Tilt angle in degrees

# Generate coordinate grid
x = np.linspace(-10, 10, N)
y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(x, y)

# Generate the Gaussian beam field at z = 0
U0 = gaussian_beam_field(X, Y, w0)

# Propagate the field to z = 1
U1 = fresnel_propagate(U0, z, lambda_, dx)

# Rotate the coordinates to simulate the tilted plane
X_rot, Y_rot = rotate_coordinates(X, Y, angle)

# Interpolate the propagated field onto the rotated coordinates
from scipy.interpolate import griddata

points = np.vstack((X.flatten(), Y.flatten())).T
U1_flat = U1.flatten()
grid_z = griddata(points, U1_flat, (X_rot, Y_rot), method='cubic')

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Gaussian Beam Field at z = 0')
plt.imshow(np.abs(U0)**2, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')

plt.subplot(1, 2, 2)
plt.title('Projection on Tilted Plane at z = 1')
plt.imshow(np.abs(grid_z)**2,  extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')

plt.show()
