## Overview

**SOMA-INR** is the official implementation accompanying the paper ["Generalizable Implicit Neural Representations via Parameterized Latent Dynamics for Baroclinic Ocean Forecasting."](https://arxiv.org/pdf/2503.21588?) This repository provides a deep learning framework for learning continuous neural representations of high-resolution ocean dynamics from the **Simulating Ocean Mesoscale Activity (SOMA)** dataset.

The proposed framework, **PINROD** (Parameterized Implicit Neural Representation with Latent ODE Dynamics), combines implicit neural representations (INRs) with parameterized latent neural ordinary differential equations (Neural ODEs) to model the spatiotemporal evolution of baroclinic ocean simulations. Unlike conventional INR-based approaches that require learning separate latent representations for each simulation, PINROD learns a shared latent dynamics model conditioned on simulation parameters, enabling accurate forecasting across unseen physical settings.

### Key Features

- 🌊 **Continuous Spatial Representation**  
  Learn grid-independent implicit neural representations of ocean state variables for arbitrary spatial resolutions.

- 🧠 **Parameterized Latent Dynamics**  
  Model latent state evolution using Neural ODEs conditioned on physical parameters, allowing the framework to generalize across different simulation configurations.

- 🚀 **Generalizable Ocean Forecasting**  
  Predict future ocean states for unseen simulation parameters without retraining individual models.

- ⚡ **Efficient Scientific Surrogate Modeling**  
  Serve as a fast surrogate for computationally expensive numerical ocean simulations, making the framework suitable for many-query applications such as uncertainty quantification, parameter studies, and inverse problems.

### Repository Contents

This repository includes implementations for:

- Data preprocessing for the SOMA dataset
- Implicit neural representation (INR) models
- Parameterized latent Neural ODE dynamics
- Training and evaluation pipelines
- Reproducible experiments from the paper

For more details, please refer to our paper:

> **Generalizable Implicit Neural Representations via Parameterized Latent Dynamics for Baroclinic Ocean Forecasting**  
> *Guang Zhao, et al.*
