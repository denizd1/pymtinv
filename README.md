# pymtinv: 2D Magnetotelluric Inversion Library

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research_prototype-orange)

**pymtinv** is a Python-based scientific computing library designed for **2D Magnetotelluric (MT) inversion**. It serves as a comprehensive research framework to benchmark classical deterministic optimization methods against novel **Probabilistic Computing (P-bit)** architectures tailored for FPGA acceleration.

This project bridges the gap between **Geophysical Inverse Problems** and **Emerging Hardware Architectures**, demonstrating how stochastic logic can revolutionize high-dimensional optimization.

---

## Key Features

- ** High-Performance Forward Solver:** Solves the Helmholtz equation for TE-mode using the Finite Difference Method (FDM) on a staggered grid.
- **∇ Adjoint State Gradient:** Implements efficient gradient calculation independent of the number of model parameters ($O(1)$ complexity), enabling large-scale inversion.
- ** Auto-Tuning Framework:**
  - **Automatic Regularization:** Determines the optimal Tikhonov parameter ($\beta$) using a fast, robust L-Curve scan.
  - **Hyperparameter Search:** Automatically tunes Learning Rate and Temperature for stochastic p-bit optimization via grid search.
- ** Probabilistic Inversion (P-bits):** Simulates **Langevin Dynamics** to emulate p-bit networks (invertible logic), enabling global search capabilities and escaping local minima.
- ** Advanced Visualization:** Plotting tools with support for non-uniform meshes (padding), logarithmic conductivity maps, and convergence graphs.
- ** FPGA Projection:** Benchmarking tools to compare CPU execution times with theoretical FPGA performance (massively parallel updates).
