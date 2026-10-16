import numpy as np
import torch
from optimizers import Adam

# 1. Initialize identical starting parameters
initial_weights = [2.0, -1.5, 0.5, 4.0]

# --- Custom Setup ---
theta_custom = np.array(initial_weights, dtype=np.float32)
custom_adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)

# --- PyTorch Setup ---
theta_torch = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
torch_adam = torch.optim.Adam([theta_torch], lr=0.1, betas=(0.9, 0.999))

print("--- Starting PyTorch Validation ---")

# 2. Run both optimizers for 5 steps with identical mock gradients
for step in range(1, 6):
    mock_grad = np.array([0.5, -0.2, 0.1, 1.0], dtype=np.float32) * step
    
    # --- Update Custom Adam ---
    theta_custom = custom_adam.update(theta_custom, mock_grad)
    
    # --- Update PyTorch Adam ---
    theta_torch.grad = torch.tensor(mock_grad, dtype=torch.float32)
    torch_adam.step() 
    torch_adam.zero_grad() 
    
    # 3. Compare the results
    torch_result = theta_torch.detach().numpy()
    max_difference = np.abs(theta_custom - torch_result).max()
    
    print(f"Step {step}: Max difference between custom and PyTorch = {max_difference:.8f}")

print("\nFinal Results:")
print("Custom Theta :", theta_custom)
print("PyTorch Theta:", theta_torch.detach().numpy())