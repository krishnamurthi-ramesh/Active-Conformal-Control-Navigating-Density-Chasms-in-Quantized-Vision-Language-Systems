# Projection Pursuit Density Ratio Estimation

This is the official implementation for the article [Projection Pursuit Density Ratio Estimation](https://openreview.net/forum?id=MgNeJO0PcF) by Meilin Wang, Wei Huang, Mingming Gong and Zheng Zhang.

## Introduction

The ppDRE method estimates high-dimensional density ratios $`r^*(\mathbf{x}) = p(\mathbf{x})/q(\mathbf{x})`$ by iteratively decomposing the problem into low-dimensional projections, forming a multiplicative projection pursuit approximation:

```math
r^*(\mathbf{x}) \approx r_K(\mathbf{x})=\prod_{k=1}^K  f_k(\mathbf{a}_k^\top \mathbf{x}).
```

Based on two independently and identically distributed (i.i.d.) samples from the two distributions, i.e. $`\{ \boldsymbol{x}_{i}^{p} \}_{i = 1}^{n_{p}} \stackrel{\text{i.i.d.}}{\sim} p(\boldsymbol{x})`$ and $`\{ \boldsymbol{x}_{i}^{q} \}_{i = 1}^{n_{q}} \stackrel{\text{i.i.d.}}{\sim} q(\boldsymbol{x})`$，the model is learned iteratively by minimizing the $`L^2`$ distance loss:

```math
\mathcal{L}(f,\mathbf{a})= \mathbb{E}_{q} \left\{r_{k-1}^2(\mathbf{x}) f^2(\mathbf{a}^\top \mathbf{x})\right\}-2\mathbb{E}_p\left\{r(\mathbf{x})f(\mathbf{a}^\top \mathbf{x})\right\}.
```
For full algorithmic details and implementation, please refer to the [paper](https://openreview.net/forum?id=MgNeJO0PcF).



## Reference

Please consider to cite our work if you find our algorithm or code useful in your research.

```
@inproceedings{
wang2025projection,
title={Projection Pursuit Density Ratio Estimation},
author={Meilin Wang and Wei Huang and Mingming Gong and Zheng Zhang},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=MgNeJO0PcF}
}
```
