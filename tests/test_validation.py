import sys
import os
import torch
import numpy as np

# Allow the test folder to see the root optimizers.py file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import Adam

def test_adam_pytorch_parity():
    # Fix the seeds for exact reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Define a starting position and a dummy gradient
    initial_theta = np.array([1.5, -2.0], dtype=np.float32)
    grad = np.array([0.5, -0.1], dtype=np.float32)

    # 1. Run Custom NumPy Adam
    custom_adam = Adam(lr=0.1, beta1=0.9, beta2=0.999)
    custom_theta = np.copy(initial_theta)
    custom_theta = custom_adam.update(custom_theta, grad)

    # 2. Run PyTorch Adam
    tensor_theta = torch.tensor(initial_theta, requires_grad=True)
    torch_adam = torch.optim.Adam([tensor_theta], lr=0.1, betas=(0.9, 0.999))
    
    tensor_theta.grad = torch.tensor(grad)
    torch_adam.step()

    # 3. Assert they are mathematically identical (tolerance 1e-5)
    np.testing.assert_allclose(
        custom_theta, 
        tensor_theta.detach().numpy(), 
        rtol=1e-5, 
        atol=1e-5,
        err_msg="NumPy Adam does not match PyTorch Adam!"
    )