# TCR–pMHC Binding Prediction

This repository contains all data processing, model training, and evaluation scripts used in our study on predicting T-cell receptor (TCR) binding to peptide–MHC (pMHC) complexes using deep learning.

## Overview

We implemented and compared several deep learning models, including:
- CNN with BLOSUM encodings
- CNN with TCRLang transformer-based embeddings
- Bidirectional LSTM and stacked LSTM models
- CNN combined with Variational Autoencoder for unsupervised denoising
- Extensive hyperparameter tuning and performance evaluation by peptide (AUC, AUC0.1)

Our goal was to assess the impact of embedding richness and data quality on TCR–pMHC interaction prediction, and to explore model robustness under noisy conditions.
