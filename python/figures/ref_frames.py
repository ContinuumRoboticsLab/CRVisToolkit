import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Arc parameters
center = (5, 0)
radius = 5
theta1 = 180  # Start angle in degrees
theta2 = 110  # End angle in degrees

# Create the arc
arc = Arc(
    center,
    2 * radius,
    2 * radius,
    theta1=theta2,
    theta2=theta1,
    lw=2,
    color="blue",
    zorder=1,
)
ax.add_patch(arc)

# Generate points along the arc for plotting
theta = np.linspace(np.radians(theta1), np.radians(theta2), 100)
x_arc = center[0] + radius * np.cos(theta)
y_arc = center[1] + radius * np.sin(theta)

# Calculate the positions of the start and end points
end_point = (
    center[0] + radius * np.cos(np.radians(theta1)),
    center[1] + radius * np.sin(np.radians(theta1)),
)
start_point = (
    center[0] + radius * np.cos(np.radians(theta2)),
    center[1] + radius * np.sin(np.radians(theta2)),
)

# Calculate the tangent vectors (derivatives) at start and end points
start_tangent = (np.sin(np.radians(theta2)), -np.cos(np.radians(theta2)))
start_normal = (-np.cos(np.radians(theta2)), -np.sin(np.radians(theta2)))
end_tangent = (-np.sin(np.radians(theta1)), -np.cos(np.radians(theta1)))
end_normal = (-np.cos(np.radians(theta1)), -np.sin(np.radians(theta1)))

# Scale for the coordinate frame arrows
arrow_scale = 1.0

# Draw coordinate frames
# Start frame
ax.arrow(
    start_point[0],
    start_point[1],
    arrow_scale * start_tangent[0],
    arrow_scale * start_tangent[1],
    color="red",
    width=0.05,
    head_width=0.2,
    head_length=0.3,
    zorder=2,
)
ax.arrow(
    start_point[0],
    start_point[1],
    arrow_scale * start_normal[0],
    arrow_scale * start_normal[1],
    color="green",
    width=0.05,
    head_width=0.2,
    head_length=0.3,
    zorder=2,
)

# End frame
ax.arrow(
    end_point[0],
    end_point[1],
    arrow_scale * end_tangent[0],
    arrow_scale * end_tangent[1],
    color="red",
    width=0.05,
    head_width=0.2,
    head_length=0.3,
    zorder=2,
)
ax.arrow(
    end_point[0],
    end_point[1],
    arrow_scale * end_normal[0],
    arrow_scale * end_normal[1],
    color="green",
    width=0.05,
    head_width=0.2,
    head_length=0.3,
    zorder=2,
)

# Add labels
label_offset = 0.5
dist_frame_x = start_point[0] - label_offset
dist_frame_z = start_point[1] + label_offset

prox_frame_x = end_point[0] - label_offset
prox_frame_z = end_point[1] - label_offset

ax.text(dist_frame_x, dist_frame_z, r"$\{\mathcal{F}_d\}$", fontsize=14)
ax.text(prox_frame_x, prox_frame_z, r"$\{\mathcal{F}_p\}$", fontsize=14)


ax_label_scale = 1.4
ax.text(
    start_point[0] + arrow_scale * start_tangent[0] * ax_label_scale,
    start_point[1] + arrow_scale * start_tangent[1] * ax_label_scale,
    r"$\hat{\mathbf{z}}_d$",
    fontsize=14,
)
ax.text(
    start_point[0] + arrow_scale * start_normal[0] * ax_label_scale,
    start_point[1] + arrow_scale * start_normal[1] * ax_label_scale,
    r"$\hat{\mathbf{x}}_d$",
    fontsize=14,
)

ax.text(
    end_point[0] + arrow_scale * end_tangent[0] * ax_label_scale - label_offset,
    end_point[1] + arrow_scale * end_tangent[1] * ax_label_scale,
    r"$\hat{\mathbf{z}}_p$",
    fontsize=14,
)
ax.text(
    end_point[0] + arrow_scale * end_normal[0] * ax_label_scale,
    end_point[1] + arrow_scale * end_normal[1] * ax_label_scale,
    r"$\hat{\mathbf{x}}_p$",
    fontsize=14,
)

# Set equal aspect ratio and limits
x_center = center[0] / 2
y_center = 3
ax.set_aspect("equal")
ax.set_xlim(x_center - 6, x_center + 6)
ax.set_ylim(y_center - 6, y_center + 6)

# Add grid, labels, and title
# ax.grid(True)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Z", fontsize=12)
ax.set_title("Circular Arc with Reference Frames", fontsize=14)

# Draw center point and center to arc lines
# ax.plot(center[0], center[1], 'ko', markersize=5)
# ax.plot([center[0], start_point[0]], [center[1], start_point[1]], 'k--', lw=1)
# ax.plot([center[0], end_point[0]], [center[1], end_point[1]], 'k--', lw=1)

plt.show()
