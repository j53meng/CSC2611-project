# Geometric Constraints on Semantic Change — A Dynamical Analysis of Dimensionality Expansion and Structural Instability in Word Embeddings

## Overview 

This repository contains the code and analysis for our final project for CSC2611: Computational Models of Semantic Change at the University of Toronto.

## Data

To run the analysis, you first need to download the required historical embeddings. We provide a shell script to automate this process.

```bash download_data.sh```

This project utilizes the [HistWords](https://github.com/williamleif/histwords/tree/master) framework. We cloned the original repository to leverage its utilities for loading embeddings and performing nearest-neighbor searches.

## Analysis
### Intrinsic Dimensionality (ID)

- Generate Scores: Use `id_utils.py` to calculate the intrinsic dimensionality scores across the temporal slices.

- Run Analysis: Execute `id_analysis.py` to reproduce the primary findings from the paper, including the identification of ID Bursts and the longitudinal correlation shifts (Predictability Index).

### Anisotropy
To investigate how the local shape of the manifold guides semantic drift:

Refer to `anisotropy.py`. This script handles both the generation of anisotropy scores and the subsequent analysis regarding semantic direction and alignment.