# GPU-Accelerated Evolutionary Optimisation

This repository accompanies the doctoral research conducted as part of my PhD thesis, which explored the optimisation and performance of Evolutionary Algorithms (EA) using modern GPU architectures.

## Overview

EAs are powerful tools for searching the large and intricate search spaces found in difficult problems. Despite their effectiveness, EAs are computationally expensive, often prematurely converge to local optima, and have practical limitations due to prolonged execution times. This repository provides:

- **Optimised GPU-based algorithms**: Implementations of Island Model Genetic Algorithm (IMGA), Differential Evolution (DE), and novel variants (DEGI, DEGIAC, DEGIACS).
- **Numerical optimisation benchmarks**: A standardised evaluation framework for comparing GPU-based DE algorithms.

## Key Features

1. **State-of-the-art GPU-accelerated EAs**:
   - Achieve speedups of up to 470 times compared to CPU implementations.
   - Adaptive algorithms (DEGIAC and DEGIACS) with self-adaptive control parameters and online learning for strategy selection.

2. **GPU Optimisation**:
   - Optimised implementations for NVIDIA GPUs using CUDA.
   - Benchmarks to evaluate performance on diverse GPU hardware.

## Access the Research

The research and findings in this repository are detailed in the accompanying PhD thesis:

**[GPU Accelerated Evolutionary Optimisation](https://hdl.handle.net/10072/433692)**

## Publications

This repository is based on research presented in the following publications:

1. **"Acceleration of Genetic Algorithm on GPU CUDA Platform"**  
   *Dylan Janssen, Alan Wee-Chung Liew.*  
   Published in *2019 20th International Conference on Parallel and Distributed Computing, Applications and Technologies (PDCAT)*  
   [DOI: 10.1109/PDCAT46702.2019.00047](https://ieeexplore.ieee.org/document/9028996)
   
2. **"Graphics processing unit acceleration of the island model genetic algorithm using the CUDA programming platform"**  
   *Dylan M. Janssen, Wayne Pullan, Alan Wee-Chung Liew.*  
   Published in *Concurrency and Computation: Practice and Experience*  
   [DOI: 10.1002/cpe.6286](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.6286)

3. **"GPU Based Differential Evolution: New Insights and Comparative Study"**  
   *Dylan Janssen, Wayne Pullan, Alan Wee-Chung Liew.*  
   Preprint available on *arXiv*  
   [arXiv:2405.16551](https://arxiv.org/abs/2405.16551v1)


## Citation

If you use this repository in your research, please cite:

```
@phdthesis{Janssen2025,
  title     = {GPU Accelerated Evolutionary Optimisation},
  author    = {Dylan Janssen},
  year      = {2024},
  school    = {Griffith University},
  url       = {https://github.com/DylanJanssen/GPU-Accelerated-Evolutionary-Optimisation}
}
```

## License

This repository is licensed under the [MIT License](LICENSE).


