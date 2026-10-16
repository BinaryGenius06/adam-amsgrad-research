import numpy as np
from optimizers import Adam, AMSGrad

if __name__ == "__main__":
    print("--- Day 4: Adam Manual Verification ---")
    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    # Initialize theta and gradients as numpy arrays for the vectorized optimizers
    theta_adam = np.array([0.0], dtype=np.float32)
    
    # Step 1
    theta_adam = adam.update(theta_adam, grad=np.array([10.0], dtype=np.float32))
    print(f"Step 1: theta = {theta_adam[0]:.6f}")
    
    # Step 2
    theta_adam = adam.update(theta_adam, grad=np.array([-1.0], dtype=np.float32))
    print(f"Step 2: theta = {theta_adam[0]:.6f}")

    print("\n--- Day 5: AMSGrad Manual Verification ---")
    amsgrad = AMSGrad(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    theta_ams = np.array([0.0], dtype=np.float32)
    
    # Step 1
    theta_ams = amsgrad.update(theta_ams, grad=np.array([10.0], dtype=np.float32))
    print(f"Step 1: theta = {theta_ams[0]:.6f}")
    
    # Step 2
    theta_ams = amsgrad.update(theta_ams, grad=np.array([-1.0], dtype=np.float32))
    print(f"Step 2: theta = {theta_ams[0]:.6f}")