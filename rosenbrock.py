import numpy as np
import matplotlib.pyplot as plt
from optimizers import SGD, Momentum, Adam, AMSGrad

# 1. Define the Rosenbrock Function and its Gradient
def rosenbrock(theta):
    x, y = theta[0], theta[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(theta):
    x, y = theta[0], theta[1]
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy], dtype=np.float32)

# 2. Simulation Runner
def run_optimizer_2d(optimizer, start_pos, steps=3000):
    theta = np.array(start_pos, dtype=np.float32)
    trajectory = [theta.copy()]
    
    for _ in range(steps):
        grad = rosenbrock_gradient(theta)
        theta = optimizer.update(theta, grad)
        trajectory.append(theta.copy())
        
    return np.array(trajectory)

# --- Set up the Experiment ---
# Start far away from the optimal minimum, which is at (1.0, 1.0)
start_pos = [-2.0, 2.0]
steps = 4000

# Notice the learning rates: The Rosenbrock gradient can be massive.
# Basic SGD needs a tiny learning rate to avoid exploding, while Adam adapts automatically.
opts = {
    'SGD': (SGD(lr=0.0001), 'black'),
    'Momentum': (Momentum(lr=0.0001, beta=0.9), 'purple'),
    'Adam': (Adam(lr=0.01, beta1=0.9, beta2=0.999), 'red'),
    'AMSGrad': (AMSGrad(lr=0.01, beta1=0.9, beta2=0.999), 'blue')
}

trajectories = {}
print("Running 2D simulations...")
for name, (opt, _) in opts.items():
    trajectories[name] = run_optimizer_2d(opt, start_pos, steps)

# --- Plot the 2D Contour Map ---
print("Generating plot...")
plt.figure(figsize=(12, 8))

# Create a grid of points for the contour background
x_grid = np.linspace(-2.5, 2.0, 400)
y_grid = np.linspace(-1.0, 3.0, 400)
X, Y = np.meshgrid(x_grid, y_grid)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# Draw the contour (using log scale so the steep walls are visible)
plt.contourf(X, Y, np.log(Z + 1e-8), levels=30, cmap='viridis', alpha=0.6)

# Overlay the trajectories
for name, traj in trajectories.items():
    color = opts[name][1]
    plt.plot(traj[:, 0], traj[:, 1], label=name, color=color, linewidth=2, alpha=0.8)
    # Mark the end point
    plt.scatter(traj[-1, 0], traj[-1, 1], color=color, s=50, edgecolors='white', zorder=5)

# Mark the starting point and optimal minimum
plt.scatter(start_pos[0], start_pos[1], color='white', marker='X', s=100, label='Start (-2, 2)')
plt.scatter(1.0, 1.0, color='gold', marker='*', s=300, edgecolors='black', label='Global Minimum (1, 1)', zorder=10)

plt.title("Optimizer Trajectories on the Rosenbrock Function")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()