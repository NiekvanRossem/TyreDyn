import numpy as np
import matplotlib.pyplot as plt

# input state
LOADED_RADIUS = 0.31

# tyre properties
UNLOADED_RADIUS          = 0.35              #Free tyre radius
WIDTH                    = 0.205             #Nominal section width of the tyre
RIM_RADIUS               = 0.2413            #Nominal rim radius
RIM_WIDTH                = 0.152             #Rim width
ASPECT_RATIO             = 0.6               #Nominal aspect ratio

# angular coordinates
theta_tyre = np.linspace(0, 2 * np.pi, 32)
theta_rim  = np.linspace(0, 2 * np.pi, 32)

# width coordinates
y_tyre = np.linspace(-WIDTH / 2, WIDTH / 2, 2)
y_rim  = np.linspace(-RIM_WIDTH / 2, RIM_WIDTH / 2, 2)

# create meshgrid
theta_tyre, y_tyre = np.meshgrid(theta_tyre, y_tyre)
theta_rim,  y_rim  = np.meshgrid(theta_rim, y_rim)

# x and z coordinates
x_tyre = UNLOADED_RADIUS * np.cos(theta_tyre)
z_tyre = UNLOADED_RADIUS * np.sin(theta_tyre) + LOADED_RADIUS
x_rim  = RIM_RADIUS * np.cos(theta_rim)
z_rim  = RIM_RADIUS * np.sin(theta_rim) + LOADED_RADIUS

# create deformation of contact patch
z_tyre = np.maximum(z_tyre, 0.0)

#left_wall =


# ground plane limits
xlim = [-0.5, 0.5]
ylim = [-0.5, 0.5]

ground_plane = np.meshgrid(np.linspace(*xlim, 10), np.linspace(*ylim, 10))
#ground_plane = np.array(ground_plane, np.zeros_like(ground_plane[0]))

# plot result
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(ground_plane[0], ground_plane[1], np.zeros_like(ground_plane[0]), color='black', linewidth=0.5)
ax.plot_surface(x_tyre, y_tyre, z_tyre, color='#444444')

ax.plot_surface(x_rim,  y_rim,  z_rim,  color='#222222')
ax.set(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], zlim=[0, 1])
ax.grid(False)
ax.set_aspect('equal')
plt.show()