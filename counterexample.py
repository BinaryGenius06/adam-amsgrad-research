import numpy as np
import matplotlib.pyplot as plt
from optimizers import Adam, AMSGrad

def get_gradient(t, C=10.0):
    # Period-3 cycle: return C every 3rd step, otherwise return -1
    # Converted to numpy arrays for the vectorized optimizers
    if t % 3 == 1:
        return np.array([C], dtype=np.float32)
    else:
        return np.array([-1.0], dtype=np.float32)

def run_counterexample(optimizer, steps=3000, C=10.0):
    theta = np.array([0.0], dtype=np.float32)
    trajectory = []

    for t in range(1, steps + 1):
        grad = get_gradient(t, C)
        theta = optimizer.update(theta, grad)
        
        # Project back into the feasible set [-1, 1] using numpy clip
        theta = np.clip(theta, -1.0, 1.0)
        
        # Store the scalar value for plotting
        trajectory.append(theta[0])

    return trajectory

if __name__ == "__main__":
    # --- Run the Experiment ---
    steps = 3000
    C_val = 10000.0

    # The mathematical trap from Reddi et al. Theorem 1
    adam_opt = Adam(lr=0.01, beta1=0.0, beta2=1.0 / (1.0 + C_val**2))
    amsgrad_opt = AMSGrad(lr=0.01, beta1=0.0, beta2=1.0 / (1.0 + C_val**2))

    print("Running Adam...")
    adam_traj = run_counterexample(adam_opt, steps=steps, C=C_val)

    print("Running AMSGrad...")
    amsgrad_traj = run_counterexample(amsgrad_opt, steps=steps, C=C_val)

    # --- Plot the Trajectories ---
    plt.figure(figsize=(10, 5))
    plt.plot(adam_traj, label='Adam', color='red', alpha=0.8)
    plt.plot(amsgrad_traj, label='AMSGrad', color='blue', alpha=0.8)

    # Add reference lines for the bounds
    plt.axhline(1.0, color='gray', linestyle='--', label='Worst point (+1)')
    plt.axhline(-1.0, color='green', linestyle='--', label='Optimal point (-1)')

    plt.xlabel('Steps (t)')
    plt.ylabel('Theta')
    plt.title(f'Adam vs AMSGrad on Period-3 Counterexample (C={C_val})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()