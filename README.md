# Deep Learning Optimizers from Scratch: Adam, AMSGrad, and the Convergence Trap

An applied algorithms research project focused on the theoretical guarantees and empirical edge cases of adaptive learning rate optimization algorithms. This repository implements standard optimizers (SGD, Momentum, Adam, AMSGrad) entirely from scratch using vectorized NumPy operations, validates them against PyTorch's C++ backend, and reproduces the famous mathematical trap from Reddi et al. (2018).

## 📌 Abstract
While Adam (Kingma & Ba, 2014) is the industry standard for deep learning optimization, its mathematical proof of convergence contains a fundamental flaw. Under specific conditions with extreme gradient variance, Adam's exponential moving average fails to retain sufficient memory of past gradients, causing it to diverge. This project empirically maps this failure zone, implements the AMSGrad correction, and benchmarks these algorithms across multi-dimensional non-convex topographies.

## 📂 Repository Structure
```text
.
├── optimizers.py                 # Vectorized core optimizer classes (SGD, Momentum, Adam, AMSGrad)
├── counterexample.py             # Reproduces the Period-3 trap from Reddi et al.
├── rosenbrock.py                 # 2D non-convex contour optimization benchmark
├── validation.py                 # Cross-examination against torch.optim.Adam
├── sweep.py                      # Hyperparameter boundary mapping
├── experiments/                  # Step-by-step manual validation scripts
│   ├── adam_verification.py
│   └── toy_test.py
├── assets/                       # Generated contour maps and trajectory plots
└── README.md