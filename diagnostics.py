import numpy as np
import matplotlib.pyplot as plt
from optimizers import Adam, AMSGrad
from counterexample import run_counterexample # Assumes this runs the loop and returns the optimizer object

# 1. Run the environments
# (Adjust these parameters to match your existing counterexample setup)
start_pos = [0.0]
steps = 3000

adam_opt = Adam(lr=0.1, beta1=0.0, beta2=0.5)
amsgrad_opt = AMSGrad(lr=0.1, beta1=0.0, beta2=0.5)

# Run the simulations (ensure run_counterexample returns the modified optimizer)
run_counterexample(adam_opt, steps)
run_counterexample(amsgrad_opt, steps)

# 2. Extract the variance histories
# We slice the first 100 steps to clearly see the initial exponential decay
steps_to_plot = 100
adam_v = [v[0] for v in adam_opt.v_history[:steps_to_plot]]
amsgrad_v = [v[0] for v in amsgrad_opt.v_history[:steps_to_plot]]

# 3. Plot the Internal Mathematical State
plt.figure(figsize=(10, 6))
plt.plot(adam_v, label="Adam ($v_t$)", color='red', linewidth=2)
plt.plot(amsgrad_v, label="AMSGrad ($\\hat{v}_t$)", color='blue', linewidth=2, linestyle='--')

plt.title("Internal Variance Tracking: The Period-3 Memory Drain", fontsize=14)
plt.xlabel("Time Step (t)", fontsize=12)
plt.ylabel("Variance Accumulator State", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Highlight the problem
plt.annotate('Adam rapidly forgets the gradient spike', 
             xy=(10, adam_v[10]), xytext=(20, max(adam_v)*0.8),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=11)

plt.tight_layout()
plt.savefig("figures/diagnostic_v_tracking.png", dpi=300)
print("Diagnostic plot saved to figures/diagnostic_v_tracking.png")
plt.show()