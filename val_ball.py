import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

point_all = []
for _ in range(10000):
    l_max = 81
    theta = np.random.uniform(0, 2 * np.pi)  # 随机生成角度 θ，范围 [0, 2π)
    phi = np.random.uniform(0, np.pi)  # 随机生成角度 φ，范围 [0, π)
    r = l_max * np.cbrt(np.random.uniform(0, 1))  # 随机生成半径 r，使用 cbrt(random) 保证均匀分布

    # Assuming self.goal is a dictionary containing the x, y, z coordinates
    goal = {'x': r * np.sin(phi) * np.cos(theta),
            'y': r * np.sin(phi) * np.sin(theta),
            'z': r * np.cos(phi)}
    point_all.append(goal)
    
# Extract x, y, z coordinates for plotting
x_coords = [point['x'] for point in point_all]
y_coords = [point['y'] for point in point_all]
z_coords = [point['z'] for point in point_all]
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the point
ax.scatter(x_coords, y_coords, z_coords, color='r')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Random Point in 3D Space')

# Show the plot
plt.show()
