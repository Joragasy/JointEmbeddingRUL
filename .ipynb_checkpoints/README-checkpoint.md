# Robust Self-Supervised Transformer for Remaining Useful Life (RUL) Estimation

This repository implements a **dual-encoder transformer architecture** for Remaining Useful Life (RUL) estimation in industrial systems. The proposed approach focuses on learning **robust, degradation-aware representations** by aligning latent embeddings extracted from **clean sensor data** and **noised / partially masked views** of the same multivariate time series.

The model combines **self-supervised representation learning** with **supervised RUL regression**, aiming to improve robustness to noise, missing data, and operational variability commonly encountered in real-world Prognostics and Health Management (PHM) applications.

---

## Overview

The architecture consists of:

- **Two transformer encoders**:
  - A **clean-data encoder** operating on original sensor time series
  - A **corrupted-data encoder** trained on noise-perturbed and partially masked inputs
- A **shared latent embedding space**, enforced through representation alignment between the two encoders
- A downstream **regression head** that predicts the Remaining Useful Life from the learned latent representation

By forcing both encoders to produce consistent embeddings, the model learns representations that are invariant to noise and partial observability while preserving degradation-relevant temporal structure.

---

## Key Features

- Dual transformer encoders with shared latent space  
- Self-supervised alignment between clean and corrupted signal representations  
- Robust feature learning under noise and missing data  
- Modular RUL regression head  
- Suitable for long-horizon multivariate time-series modeling  

---

## Target Applications

- Prognostics and Health Management (PHM)  
- Predictive maintenance  
- Industrial asset health monitoring  
- Run-to-failure datasets (e.g., CMAPSS, bearing datasets, battery degradation data)  

---

## Research Motivation

Accurate RUL estimation is challenged by sensor noise, missing values, and varying operating conditions. Most existing transformer-based RUL models rely purely on supervised learning and clean inputs. This project investigates a **robust self-supervised learning strategy** that improves generalization and stability by explicitly aligning clean and corrupted views of degradation signals.

---

## Status

This repository is intended for **research and experimentation**. The architecture is designed to be extensible and can be adapted for different datasets, masking strategies, and alignment objectives.

---
