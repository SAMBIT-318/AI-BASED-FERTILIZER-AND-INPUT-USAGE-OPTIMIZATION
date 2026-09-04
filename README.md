# AI-Based Fertilizer and Input Usage Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, data-driven precision agriculture decision-support system designed to replace inefficient blanket chemical application with parcel-specific, variable-rate nutrient dosing. The framework couples tabular machine learning models (XGBoost, Random Forest) with constrained linear programming (PuLP/CBC) to maximize harvest yields and Benefit-to-Cost Ratios (BCR) within strict farmer budget limitations while mitigating groundwater and environmental leaching.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Architecture](#dataset-architecture)
- [Methodology & Mathematical Formulation](#methodology--mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Local Setup & Installation](#local-setup--installation)
- [Free Cloud Deployment](#free-cloud-deployment)
- [Project Viva & Academic Credits](#project-viva--academic-credits)

---

## Problem Statement
Conventional agriculture heavily relies on uniform, blanket fertilizer dispersal across entire fields without accounting for spatial variations in soil chemistry or crop phenological stages:
* **Overuse:** Leads to soil acidification, groundwater nitrate leaching, eutrophication in aquatic bodies, and inflated input expenses.
* **Underuse:** Causes severe yield penalties and substantial financial losses for cultivators.
* **Core Challenge:** Smallholder farmers lack parcel-specific advisory systems aligned with dynamic soil parameters (N, P, K, pH, moisture) and bounded capital constraints.

---

## Key Features
* **Parcel-Specific Predictive Dosing:** Accurately forecasts site-specific macronutrient (N-P-K) deficits and expected harvest yields using real-time soil test metrics and climatic data.
* **Budget-Constrained Linear Optimization:** Formulates and solves bounded linear programming models using **PuLP (CBC Solver)** to determine the optimal blend of commercial and organic fertilizers (Urea, DAP, MOP, Complex 14-35-14, Compost) without exceeding wallet caps.
* **Environmental Leaching Guards:** Implements upper-bound uptake constraints to curb excess chemical application by **15% to 30%**, protecting groundwater tables and long-term soil structure.
* **Interactive Web Platform:** Built with **Streamlit**, featuring real-time soil-test sliders, nutrient balance tracking, exploratory dataset heatmaps, and downloadable fertilizer dosage schedules.

---

## System Architecture

```text
[ Soil Test Kit / IoT Sensors ]      [ Weather Forecasts / Climate Data ]
   (N, P, K, pH, Soil Moisture)           (Temp, Humidity, Rainfall)
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Data Preprocessing Layer     │
                    │  (Standardization, Categorical  │
                    │   Encoding & Outlier Triage)    │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Machine Learning Inference Engine│
                    │   - XGBoost / Random Forest     │
                    │   - Yield & Crop Deficit Models │
                    └────────────────┬────────────────┘
                                     ▼
                         Nutrient Deficits (N, P, K)
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  PuLP Linear Programming Engine  │
                    │   - Objective: Min Input Cost   │
                    │   - Subject to: Wallet Limit    │
                    │   - Guard: Max Leaching Bounds  │
                    └────────────────┬────────────────┘
                                     ▼
                    ┌─────────────────────────────────┐
                    │  Streamlit Application Layer    │
                    │   - Variable-Rate NPK Dosage    │
                    │   - Budget Impact & Analytics   │
                    └─────────────────────────────────┘
