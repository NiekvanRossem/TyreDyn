import numpy as np
import matplotlib.pyplot as plt

def generate_tyre_and_rim(
    tyre_radius=0.35,
    rim_radius=0.22,
    tyre_width=0.25,
    rim_width=0.20,
    n_theta=120,
    n_width=60,
):
    # -----------------
    # Angular sweep
    # -----------------
    theta = np.linspace(0, 2*np.pi, n_theta)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # -----------------
    # Rim (cylindrical)
    # -----------------
    z_rim = np.linspace(-rim_width/2, rim_width/2, n_width)
    Theta_rim, Z_rim = np.meshgrid(theta, z_rim, indexing="ij")

    X_rim = rim_radius * np.cos(Theta_rim)
    Y_rim = rim_radius * np.sin(Theta_rim)

    # -----------------
    # Tyre profile (width direction)
    # -----------------
    z_tyre = np.linspace(-tyre_width/2, tyre_width/2, n_width)
    u = z_tyre / (tyre_width / 2)        # [-1, 1]

    # Sidewall curvature (smooth, round)
    sidewall = np.sqrt(np.clip(1 - u**2, 0, 1))

    # Radial profile from rim → tread
    r_profile = rim_radius + (tyre_radius - rim_radius) * sidewall

    # Expand to full surface: (theta, width)
    R_tyre = np.tile(r_profile, (n_theta, 1))
    Z_tyre = np.tile(z_tyre, (n_theta, 1))

    # -----------------
    # Sweep around axis
    # -----------------
    X_tyre = R_tyre * cos_t[:, None]
    Y_tyre = R_tyre * sin_t[:, None]

    return (X_rim, Y_rim, Z_rim), (X_tyre, Y_tyre, Z_tyre)


# =====================
# Plot
# =====================
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

rim, tyre = generate_tyre_and_rim()

Xr, Yr, Zr = rim
Xt, Yt, Zt = tyre

ax.plot_wireframe(Xr, Yr, Zr, color="#666666", linewidth=0.5)
ax.plot_wireframe(Xt, Yt, Zt, color="#444444", linewidth=0.5)

ax.set_box_aspect([1, 1, 0.6])
ax.set_axis_off()
ax.view_init(elev=20, azim=45)

plt.show()
