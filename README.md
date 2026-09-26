# Smart Kishan : AI-Based Fertilizer and Input Usage Optimization

Smart Kishan is an AI-powered agricultural control center designed to replace farming guesswork with data-driven precision agronomy. It provides automated 4R fertilizer optimization, live optical soil analysis, computer-vision plant pathology, and predictive analytics for smart irrigation and crop market prices.

---

## 🌟 Key Features

1. **4R Nutrient Stewardship & Cost Optimization**
   - Implements linear programming algorithms via SciPy's HiGHS solver.
   - Strictly enforces commercial purchase budgets while meeting soil elemental requirements.
   - Formulates split application schedules across Basal Dressing, Vegetative Growth, and Flowering.

2. **Optical Soil Triage & Disease Diagnostics**
   - Computer vision analysis of soil surface color and texture to estimate soil properties.
   - Leaf pathology diagnosis providing disease detection, pest risk indices, and remedial sprays.

3. **Predictive Agronomic Suite**
   - **Crop Recommender:** Ensemble ExtraTrees Classifier predicting optimal crops from environmental indicators.
   - **Fertilizer Classifier:** Machine learning selection of ideal chemical blends.
   - **Yield Regressor:** Anticipates total production (t/acre) based on balanced NPK parameters.
   - **Smart Irrigation & Pest Probability:** Predicts millimeter water demands and disease risk.
   - **Market Mandi Forecasting:** Projects harvest commodity prices.

4. **Multi-Role Access & User Support**
   - Separate access pipelines for Farmers and Authorized Administrators.
   - Instant direct self-service password reset.
   - Help request triage, live review management with admin replies, and audit logs.

5. **Certified Digital Prescription Cards**
   - Generates official, downloadable PDF dossiers with ReportLab.

---

## 🚀 Setup & Installation

### 1. Clone & Prepare Environment
```bash
git clone [https://github.com/your-repo/smart-kishan.git](https://github.com/your-repo/smart-kishan.git)
cd smart-kishan
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
# AI-Based Fertilizer and Input Usage Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Database: Supabase](https://img.shields.io/badge/Database-Supabase%20PostgreSQL-3ECF8E.svg)](https://supabase.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, data-driven precision agriculture decision-support system designed to replace inefficient blanket chemical application with parcel-specific, variable-rate nutrient dosing. The framework couples robust tabular ensemble models (**Random Forest**) and optical plant diagnosis with constrained mathematical optimization (**SciPy Linear Programming**) to maximize harvest yields and Benefit-to-Cost Ratios (BCR) within strict farmer budget limits while mitigating groundwater and environmental leaching.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Mathematical Formulation & Optimization](#mathematical-formulation--optimization)
- [Repository Structure](#repository-structure)
- [Local Setup & Installation](#local-setup--installation)
- [Cloud Deployment & Secrets](#cloud-deployment--secrets)
- [Academic Credits & Contributions](#academic-credits--contributions)

---

## Problem Statement
Conventional agriculture relies heavily on uniform, blanket fertilizer dispersal across entire fields without accounting for spatial variations in soil chemistry or crop phenological stages:
* **Overuse:** Leads to soil acidification, groundwater nitrate leaching, aquatic eutrophication, and inflated input expenses.
* **Underuse:** Causes severe yield penalties and financial losses for smallholder farmers.
* **Technology Gap:** Farmers frequently lack parcel-specific, vernacular advisories that align with regional land measurement units (Guntha, Decimal, Acre) and bounded capital constraints.

---

## Key Features

* **Multilingual Farmer Interface:** Full application localization supporting **English**, **हिन्दी (Hindi)**, and **ଓଡ଼ିଆ (Odia)** to ensure grassroots accessibility.
* **Dual Operational Workflows:**
  1. **Full Optimization Pipeline:** Multi-step guided workflow covering soil health diagnostics, nutrient mapping, and variable-rate fertilizer allocation.
  2. **Direct Disease & Pest Diagnostic:** Fast-path optical analysis for immediate crop symptom triage without running the full soil test pipeline.
* **Real-Time GPS Tracking & Interactive Mapping:** Captures device coordinates (Latitude & Longitude) to map the parcel and auto-calibrate regional soil baseline characteristics (e.g., Gangetic Alluvial vs. Eastern Red/Laterite zones).
* **Multi-Unit Land Converter & Acreage Breakdown:** Dynamic conversion and calculation table supporting **Acre**, **Hectare**, **Guntha**, **Decimal/Cent**, and **Square Feet**.
* **Optical Disease & Pest Identification Engine:** Analyzes leaf, stem, or pest imagery captured via camera or upload, predicts pathogen strains, estimates survival/recovery probability, and dispenses targeted chemical/biological remediation spray prescriptions.
* **Budget-Constrained Linear Optimization:** Solves bounded optimization models using **SciPy Linear Programming** to prescribe exact 50kg bag quantities of Urea, DAP, MOP, Complex (14-35-14), and Organic Compost without exceeding expenditure caps.
* **4R Nutrient Stewardship & Split Application:** Generates phased application schedules across Basal Dressing, Tillering/Vegetative, and Panicle Initiation stages to optimize Nutrient Use Efficiency (NUE).
* **Cloud Persistence & Feedback Loop:** Authenticated via **Supabase PostgreSQL** pooler connections, allowing farmers to store historical profiles and submit feedback on prescription accuracy.

---

## System Architecture

```text
  [ Real-Time GPS / Map ]        [ Optical Camera / Photo ]       [ Soil Test Card / IoT ]
  (Latitude, Longitude)           (Leaf / Pest Imagery)           (N, P, K, pH, Moisture)
             │                               │                                │
             └───────────────────────┬───────┴────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Data Preprocessing Layer    │
                    │  - Dynamic Multi-Unit Converter │
                    │  - Geo-Climatic Zone Baseline   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │ Machine Learning Inference Hub  │
                    │  - Random Forest Classifiers    │
                    │  - Random Forest Regressor      │
                    │  - Optical Pathogen & Pest Triage│
                    └────────────────┬────────────────┘
                                     │
                          Nutrient Deficits (N, P, K)
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   SciPy Linear Programming      │
                    │  - Objective: Min Fertilizer Cost│
                    │  - Constraint: Budget Ceiling   │
                    │  - Guard: Soil Leaching Limits  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Streamlit Web Interface      │
                    │  - Multilingual (EN, HI, OD)    │
                    │  - Exact 50kg Bag Counts        │
                    │  - Phased Split Timetable       │
                    │  - Exportable Dossier Receipt   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │   Supabase PostgreSQL Engine    │
                    │  - Secure User Authentication   │
                    │  - Farmer Feedback Records      │
                    └─────────────────────────────────┘
