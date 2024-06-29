import matplotlib.pyplot as plt
import numpy as np
import autograd.numpy as anp
from autograd import grad


# various parameters

w0 = 1
lambda_ = 1
k = 2 * np.pi / lambda_
zR = np.pi * w0 ** 2 / lambda_


def w(w0, z, zR):
    return w0 * (1 + (z / zR) ** 2) ** 0.5


z = 1
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
    FX_rot = FX * np.cos(theta) + FY * np.sin(theta)
    z = -FX * np.sin(theta) + z * np.cos(theta)
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
plt.title("output vortex beam required")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# grid
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
    k = 2 * anp.pi / lambda_
    fx = anp.fft.fftfreq(N, d=dx)
    theta = anp.radians(angle)
    FX, FY = anp.meshgrid(fx, fx)
    H = anp.exp(-1j * anp.pi * lambda_ * z * (FX ** 2 + FY ** 2))
    U1 = anp.fft.ifft2(anp.fft.fft2(U) * H)
    return U1


def loss(phase):
    r0 = fresnel_propagate(anp.exp(1j * phase) * reflectance, z0, 1, 0.01)
    out2 = anp.abs(r0) ** 2
    return anp.sum(anp.abs(out2 - anp.abs(output_) ** 2) ** 2)


def optimization(f, inphase, num=1000, step=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = anp.zeros_like(inphase)
    v = anp.zeros_like(inphase)
    t = 0
    phase = inphase
    loss_grad = grad(f)

    for k in range(num):
        t += 1
        s = loss_grad(phase)

        m = beta1 * m + (1 - beta1) * s
        v = beta2 * v + (1 - beta2) * (s ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        phase = phase - step * m_hat / (anp.sqrt(v_hat) + epsilon)
        if k % 100 == 0:
            print(f"Iteration{k}, loss:{f(phase)}")

    return phase


def gerchberg_saxton(target_intensity, E0, initial_phase, reflectance, convergence_limit, z, iterations=100,
                     lambda_=1.0, dx=0.01):
    amplitude = np.sqrt(target_intensity)
    field_hra = reflectance * np.exp(1j * initial_phase) * E0

    for i in range(iterations):
        field_fp = fresnel_propagate(field_hra, z, lambda_, dx)
        phase_fp = np.angle(field_fp)
        fff = amplitude * np.exp(1j * phase_fp)
        field_ = fresnel_propagate(fff, -z, lambda_, dx)
        phase_hra = np.angle(field_)
        field_hra = reflectance * np.exp(1j * phase_hra)
        current_intensity = np.abs(field_fp) ** 2
        max_relative_error = np.max(np.abs(current_intensity - target_intensity) / target_intensity)
        if max_relative_error < convergence_limit:
            print(f"Converged in {i + 1} iterations with max relative error: {max_relative_error}")
            break
    finalphase = np.angle(field_hra)
    return finalphase


th = np.random.rand(512, 512)
reflectance = np.ones((512, 512)) * 1
convergence_limit = 1e-6
z0 = 1
final_phase0 = gerchberg_saxton(np.abs(output_) ** 2, input0, th, reflectance, convergence_limit, z0, 100)
plt.imshow(np.abs(final_phase0), cmap='gray')
plt.title(f"phase mask for oblique incidence (R=1)  ( {angle} ) before optimization")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
r0 = fresnel_propagate(anp.exp(1j * final_phase0) * reflectance, z0, 1, 0.01)
out2 = anp.abs(r0) ** 2
plt.imshow(out2, cmap="inferno")
plt.title(f"output vortex in oblique  ( {angle} )")
plt.show()

optimized_phase = optimization(loss, final_phase0, 200)
plt.imshow(np.abs(optimized_phase), cmap='gray')
plt.title(f"phase mask for oblique incidence (R=1)  ( {angle} ) after optimization")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
finalim = fresnel_propagate(anp.exp(1j * optimized_phase) * reflectance, z0, 1, 0.01)
out2 = anp.abs(finalim) ** 2
plt.imshow(out2, cmap="inferno")
plt.title(f"output vortex in oblique  ( {angle} ) after optimizing")
plt.show()
