import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import SGD, Momentum, Adam, AMSGrad
import numpy as np


# Outer loop to test on the toy function f(theta) = theta^2.
def test_toy_problem(optimizer, opt_name, steps=50):
    # Initialize theta as a numpy array to match the new vectorized optimizers
    theta = np.array([10.0], dtype=np.float32) 
    print(f"--- Running {opt_name} ---")
    
    for t in range(1, steps + 1):
        # Calculate gradient (derivative of theta^2 is 2*theta)
        grad = 2 * theta
        
        # Pass current theta and grad, overwrite with the new one
        theta = optimizer.update(theta, grad)
        
        # Print progress every 10 steps to see if it converges to 0
        if t % 10 == 0:
            print(f"step {t:2}: theta = {theta[0]:.6f}")
    print()

if __name__ == "__main__":
    # --- Day 3: Testing SGD and Momentum ---
    sgd = SGD(lr=0.1)
    test_toy_problem(sgd, "SGD")

    mom = Momentum(lr=0.1, beta=0.9)
    test_toy_problem(mom, "Momentum")

    # --- Day 4 & 5: Manual Step Verification ---
    print("--- Adam Manual Verification ---")
    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    theta_adam = np.array([0.0], dtype=np.float32)
    
    theta_adam = adam.update(theta_adam, grad=np.array([10.0], dtype=np.float32))
    print(f"Step 1: theta = {theta_adam[0]:.6f}")
    
    theta_adam = adam.update(theta_adam, grad=np.array([-1.0], dtype=np.float32))
    print(f"Step 2: theta = {theta_adam[0]:.6f}")