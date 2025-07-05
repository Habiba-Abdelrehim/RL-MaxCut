# RL for Max-Cut: Deterministic Policy Gradient for RL-Based MaxCut on Sparse Ising Graphs

PyTorch reproduction of the Deterministic REINFORCE paper by Lu & Liu (2023) for Max-Cut on sparse Ising graphs, extended with a Metropolis–Hastings burn-in and greedy local search. Reaches 89–96% of the paper’s Optsicom and GSet benchmarks in under an hour per graph.

A short video presentation summarizing the research project:


https://github.com/user-attachments/assets/dc3aa06d-9b36-4a2d-975d-8a5a252da9e9




## Overview

This project implements and extends the **Deterministic REINFORCE algorithm** proposed by Lu & Liu (2023) for solving the NP-hard **MaxCut** problem on sparse Ising graphs. The MaxCut problem seeks to partition the nodes of a graph to maximize the number of edges between partitions, with applications in physics, network design, and combinatorial optimization.

Our approach introduces:

- A Metropolis–Hastings burn-in to escape shallow local minima and accelerate convergence  
- A greedy local search after policy actions to immediately improve solution quality  
- Full PyTorch implementation with systematic hyperparameter tuning  
- All experiments run in a Jupyter Notebook with GPU support via Google Colab   

---

## Key Features

- Deterministic policy-gradient reinforcement learning for MaxCut  
- Feedforward MLP with two hidden ReLU layers and a Sigmoid output layer  
- Parallel MCMC sampling for efficient state-space exploration  
- Metropolis–Hastings burn-in for improved initialization  
- Greedy local search for further energy minimization  
- Benchmarked on standard **Optsicom** and **GSet** datasets  
- Achieves competitive MaxCut scores with reduced runtime  

---

## Methodology

The project is based on the antiferromagnetic Ising model formulation of MaxCut. The deterministic REINFORCE algorithm trains a neural policy to strategically choose spin flips, bypassing the need to exhaustively explore all configurations.

### Enhancements Over the Original Paper

- Explicit architectural choices for the MLP (due to missing specifications in Lu & Liu)  
- Systematic hyperparameter search per graph instance  
- Metropolis–Hastings burn-in to improve convergence  
- Partial greedy local search for local energy minimization  

---

## Results Summary

| Dataset       | Our Method (Det. REINFORCE + Extensions) | Lu & Liu (2023) Benchmarks | Notes                          |
|---------------|-------------------------------------------|-----------------------------|--------------------------------|
| **Optsicom** (Dense) | 89–96% of MaxCut benchmarks          | 100%                        | Achieved in under 1 hour per graph |
| **GSet** (Sparse)    | 77–89% of MaxCut benchmarks          | 100%                        | Reduced training time, fewer compute resources |

See `REINFORCE_Ising_MaxCut_Report.pdf` for detailed results, tables, and hyperparameter configurations.

---

## Getting Started

### Dependencies

- Python 3.x  
- PyTorch  
- NumPy  
- Matplotlib  
- tqdm  
- Google Colab (recommended for GPU acceleration)  

### Usage

The project is provided as a Jupyter Notebook. Run the notebook interactively in Google Colab or locally with GPU support.

1. Open the notebook `REINFORCE_Ising_MaxCut.ipynb`  
2. Run the first cell to automatically download the graph datasets  
3. Select the graph instance by modifying the `file_name` variable  
4. Set hyperparameters in the corresponding lists  
5. Execute the training loop and observe the results  

---

## Datasets

This project uses:

- **Optsicom benchmark instances** ([GRAFO Research Group](https://grafo.etsii.urjc.es/optsicom/#instances))  
- **GSet sparse graph collection** ([Yinyu Ye, Stanford](https://web.stanford.edu/~yyye/yyye/Gset/))  

Datasets are automatically downloaded by the notebook.

---

## References

- Yicheng Lu & Xiao-Yang Liu (2023), *Reinforcement Learning for Ising Model*  
  [Paper Link](https://ml4physicalsciences.github.io/2023/files/NeurIPS_ML4PS_2023_248.pdf)  
- Irwan Bello et al. (2016), *Neural Combinatorial Optimization with Reinforcement Learning*  
- Elias Khalil et al. (2017), *Learning Combinatorial Optimization Algorithms Over Graphs*  

---

## Contributors

- Sarah Ameur  
- Fadi Younes  
- Habiba Abdelrehim  

School of Computer Science, McGill University  

