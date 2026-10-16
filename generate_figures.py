import numpy as np
import matplotlib.pyplot as plt
import os
from optimizers import Adam, AMSGrad

# Ensure the figures directory exists
os.makedirs("figures", exist_ok=True)

steps = 500
adam = Adam(lr=0.8, beta1=0.9, beta2=0.5)
amsgrad = AMSGrad(lr=0.8, beta1=0.9, beta2=0.5)

x_adam, x_ams = 0.0, 0.0
adam_x_hist, ams_x_hist = [], []
adam_step_hist, ams_step_hist = [], []

for t in range(1, steps + 1):
    # Period-3 environment: +10 gradient every 3rd step, -1 otherwise
    grad = 10.0 if t % 10 == 1 else -1.0
    
    # Update coordinates
    x_adam = adam.update(x_adam, grad)
    x_ams = amsgrad.update(x_ams, grad)
    
    # Clip coordinates to the [-1, 1] domain
    x_adam = np.clip(x_adam, -1.0, 1.0)
    x_ams = np.clip(x_ams, -1.0, 1.0)
    
    adam_x_hist.append(x_adam)
    ams_x_hist.append(x_ams)
    
    # Calculate effective step size: alpha / sqrt(variance)
    adam_step_hist.append(0.8 / (np.sqrt(adam.v_history[-1]) + 1e-8))
    ams_step_hist.append(0.8 / (np.sqrt(amsgrad.v_history[-1]) + 1e-8))

# ---------------------------------------------------------
# Plot 1: Parameter Trajectory
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(adam_x_hist, label="Adam", color='red')
plt.plot(ams_x_hist, label="AMSGrad", color='blue', linestyle='--')
plt.axhline(-1.0, color='black', linestyle=':', label='Optimal $x^*$')
plt.title("Parameter Trajectory ($x_t$)", fontsize=14)
plt.xlabel("Time Step (t)", fontsize=12)
plt.ylabel("Coordinate Position", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("figures/trajectory.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Effective Learning Rate
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(adam_step_hist[:100], label="Adam Effective LR", color='red')
plt.plot(ams_step_hist[:100], label="AMSGrad Effective LR", color='blue', linestyle='--')
plt.title("Effective Step Size Inflation (First 100 Steps)", fontsize=14)
plt.xlabel("Time Step (t)", fontsize=12)
plt.ylabel("$\\alpha_t / \\sqrt{\\hat{v}_t}$", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("figures/effective_lr.png", dpi=300)
plt.close()

print("Successfully generated trajectory.png and effective_lr.png in the figures/ folder.")