import matplotlib.pyplot as plt
import numpy as np

# various parameters

w0 = 1
lambda_ = 1
k = 2 * np.pi / lambda_
zR = np.pi * w0 ** 2 / lambda_


def w(w0, z, zR):
    return w0 * (1 + (z / zR) ** 2) ** 0.5


z=1
# choose a specific value of z
wZ = w(w0, z, zR)

# grid
x0 = np.linspace(-3, 3, 512)
X, Y = np.meshgrid(x0, x0)
r = np.sqrt(X ** 2 + Y ** 2)
theta = np.arctan2(Y, X)

z = np.ones(512)
def rotate(U, z, lambda_, dx, angle=45):
    N = U.shape[0]
    k = 2 * np.pi / lambda_
    fx = np.fft.fftfreq(N, d=dx)
    theta = np.radians(angle)
    FX, FY = np.meshgrid(fx, fx)
    # Consider the angle in the phase factor
    # H = np.exp(-1j * np.pi * lambda_ * z * (FX ** 2 + FY ** 2))
    FX_rot = FX * np.cos(theta) + FY * np.sin(theta)
    # FY_rot = -FX * np.sin(theta) + FY * np.cos(theta)
    z = -FX * np.sin(theta) + z*np.cos(theta)
    H = np.exp(-1j * np.pi * lambda_ * z * (FX_rot ** 2 + FY ** 2))
    U1 = np.fft.ifft2(np.fft.fft2(U) * H)
    return U1


# Gaussian beam profile(l=0,p=0)
dx = 0.01  # Sampling interval
LG_0_0 = np.sqrt(2 / np.pi) * (1 / wZ) * np.exp(-r ** 2 / wZ ** 2)
angle = 30
input0 = rotate(LG_0_0, z, lambda_, dx, angle)
input1 = LG_0_0
plt.imshow(np.abs(input0) ** 2, extent=(-2, 2, -2, 2), cmap="inferno")
plt.title("input gaussian beam")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# vortex beam (l=1,p=0)
LG_1_0 = np.sqrt(2 / np.pi) * (1 / wZ) * (np.sqrt(2) * r / wZ) * np.exp(-r ** 2 / wZ ** 2) * np.exp(1j * theta)
output_ = LG_1_0
plt.imshow(np.abs(output_) ** 2, extent=(-2, 2, -2, 2), cmap="inferno")
plt.title("output vortex beam")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# grid
# x0 = np.linspace(-255.5, 255.5, 512)
# X, Y = np.meshgrid(x0, x0)
S = np.zeros((512, 512))
h = X ** 2 + Y ** 2
for i in range(512):
    for j in range(512):
        if h[i][j] <= (3 / 2) ** 2:
            S[i][j] = 1
plt.imshow(S, cmap='gray')
plt.title('support function')
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


# Fresnel-Kirchhoff propagator function
def fresnel_propagate(U, z, lambda_, dx, angle=45):
    N = U.shape[0]
    k = 2 * np.pi / lambda_
    fx = np.fft.fftfreq(N, d=dx)
    theta = np.radians(angle)
    FX, FY = np.meshgrid(fx, fx)
    # Consider the angle in the phase factor
    # H = np.exp(-1j * np.pi * lambda_ * z * (FX ** 2 + FY ** 2))
    # FX_rot = FX * np.cos(theta) + FY * np.sin(theta)
    # FY_rot = -FX * np.sin(theta) + FY * np.cos(theta)
    # z = -FX * np.sin(theta) + z*np.cos(theta)
    H = np.exp(-1j * np.pi * lambda_ * z * (FX ** 2 + FY ** 2))
    U1 = np.fft.ifft2(np.fft.fft2(U) * H)
    return U1


def gerchberg_saxton(target_intensity, E0, initial_phase, reflectance, convergence_limit, z, iterations=100,
                     lambda_=1.0, dx=0.01):
    # Initialize variables
    amplitude = np.sqrt(target_intensity)
    # Initial guess for the wave field in the HRA plan
    field_hra = reflectance * np.exp(1j * initial_phase) * E0

    for i in range(iterations):
        # Forward propagation to the focal plane using Fresnel-Kirchhoff integral
        field_fp = fresnel_propagate(field_hra, z, lambda_, dx)

        # Replace the amplitude in the focal plane with the desired amplitude
        phase_fp = np.angle(field_fp)
        fff = amplitude * np.exp(1j * phase_fp)

        # Backward propagation to the HRA plane
        field_ = fresnel_propagate(fff, -z, lambda_, dx)

        # Update the phase in the HRA plane while keeping the amplitude and applying reflectance
        phase_hra = np.angle(field_)
        field_hra = reflectance * np.exp(1j * phase_hra)
        # if (i % 50) == 0:
        #     plt.imshow(np.abs(field_fp) ** 2, cmap="inferno")
        #     plt.title(f"iteration {i} intensity distribution")
        #     cbar = plt.colorbar()
        #     cbar.set_label("values")
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Y-axis')
        #     plt.show()
        #     plt.imshow(np.abs(S * phase_hra), cmap="gray")
        #     plt.title(f"iteration {i} phase")
        #     cbar = plt.colorbar()
        #     cbar.set_label("values")
        #     plt.xlabel('X-axis')
        #     plt.ylabel('Y-axis')
        #     plt.show()
        # Calculate the maximal relative error
        current_intensity = np.abs(field_fp) ** 2
        max_relative_error = np.max(np.abs(current_intensity - target_intensity) / target_intensity)
        # Check for convergence
        if max_relative_error < convergence_limit:
            print(f"Converged in {i + 1} iterations with max relative error: {max_relative_error}")
            break
    # Final phase distribution in the HRA plane
    finalphase = np.angle(field_hra)
    return finalphase


# Parameters
th = np.random.rand(512, 512)  # random initial phase
# Example reflectance (example: uniform reflectance of 0.9)
reflectance = np.ones((512, 512)) * 1
convergence_limit = 1e-6
z0 = 1
final_phase0 = gerchberg_saxton(np.abs(output_) ** 2, input0, th, reflectance, convergence_limit, z0, 100)
plt.imshow(np.abs(S * final_phase0), cmap='gray')
plt.title(f"phase mask for oblique incidence (R=1)  ( {angle} )")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
r0 = fresnel_propagate(np.exp(1j * final_phase0) * reflectance, z0, 1, 0.01)
out2 = np.abs(r0) ** 2
plt.imshow(out2, cmap="inferno")
plt.title(f"output vortex in oblique  ( {angle} )")
plt.show()
# # normal incidence
# final_phase1 = gerchberg_saxton(np.abs(output_) ** 2, input1, th, reflectance, convergence_limit, z0, 100)
# plt.imshow(np.abs(S * final_phase1), cmap='gray')
# plt.title("phase mask for normal incidence (R=1)")
# cbar = plt.colorbar()
# cbar.set_label("values")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# r1 = fresnel_propagate(np.exp(1j * final_phase1) * reflectance, z0, 1, 0.01)
# out1 = np.abs(r1) ** 2
# plt.imshow(out1, cmap="inferno")
# plt.title("vortex for normal incidence(R=1)")
# cbar = plt.colorbar()
# cbar.set_label("values")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# difference = np.abs(S*(final_phase0 - final_phase1))
# plt.imshow(difference, cmap="inferno")
# plt.title(f"difference in phase mask for angle {angle}")
# cbar = plt.colorbar()
# cbar.set_label("values")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()


Z = final_phase0

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(X, Y, Z, cmap='inferno')
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
# ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap='inferno')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot with Contour Plot Underneath')
plt.show()
