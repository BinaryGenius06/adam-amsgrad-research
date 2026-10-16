import matplotlib.pyplot as plt
from optimizers import Adam, get_gradient, run_counterexample

# Define the grid of parameters to test
beta1_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
# Testing standard values alongside extremely small values to catch the failure zone
beta2_values = [1e-9, 1e-7, 1e-5, 1e-3, 0.1, 0.5, 0.9, 0.99]
C_val = 10000.0
steps = 2000

# Store coordinates for plotting
success_x, success_y = [], []
failure_x, failure_y = [], []

print("Running Adam hyperparameter sweep...")

for b1 in beta1_values:
    for b2 in beta2_values:
        # Instantiate a fresh Adam optimizer for each pair
        opt = Adam(lr=0.01, beta1=b1, beta2=b2)
        
        # Run the simulation
        trajectory = run_counterexample(opt, steps=steps, C=C_val)
        final_theta = trajectory[-1]
        
        # Classify the result based on where it settled
        if final_theta > 0.0:
            failure_x.append(b1)
            failure_y.append(b2)
        else:
            success_x.append(b1)
            success_y.append(b2)

# --- Plot the Map ---
plt.figure(figsize=(8, 6))

# Plot successes and failures
plt.scatter(success_x, success_y, color='green', label='Success (Converged to -1)', s=100, marker='o')
plt.scatter(failure_x, failure_y, color='red', label='Failure (Diverged to +1)', s=100, marker='x')

plt.xlabel('Beta 1 (First Moment Decay)')
plt.ylabel('Beta 2 (Second Moment Decay)')
plt.title(f'Adam Convergence Map on Period-3 Counterexample (C={C_val})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()