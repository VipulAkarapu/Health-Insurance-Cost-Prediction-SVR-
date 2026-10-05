# Health-Insurance-Cost-Prediction-SVR-
Support Vector Regression (SVR) model in Python for estimating medical insurance charges based on individual demographic and health metrics

# Non-Linear Health Insurance Cost Prediction via SVR

High-precision Support Vector Regression (SVR) pipeline implemented in Python to model non-linear relationships between health metrics (BMI, age, smoking status) and annual medical expenses.

## Project Overview

This repository demonstrates the application of Kernelized Support Vector Machines to non-linear regression problems. By leveraging the Radial Basis Function (RBF) kernel, the model maps complex, non-linear feature interactions into higher-dimensional spaces to accurately predict insurance cost trajectories.

## Key Engineering Highlights

* **RBF Kernel SVR Optimization:** Formulated an epsilon-insensitive regression boundary ($\varepsilon$-SVR) to model multi-variate continuous targets.
* **Dual-Scaler Normalization Pipeline:** Applied Z-score standardization (`StandardScaler`) independently across feature matrices and target domains to prevent spatial distortion in $L_2$ metric computations.
* **Data Leakage Mitigation:** Enforced absolute segregation of training and testing parameter distributions during preprocessing transformation phases.
* **Dense Continuous Mesh Rendering:** Generated smooth high-resolution continuous prediction curves using continuous interval evaluation grids ($0.1$ step resolution).

## Dataset Schema

The model utilizes $1,338$ records across $7$ primary medical insurance parameters:

| Feature | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Primary beneficiary age |
| `sex` | Categorical | Gender (`female`, `male`) |
| `bmi` | Continuous | Body Mass Index ($\text{kg/m}^2$) |
| `children` | Integer | Number of dependents covered |
| `smoker` | Categorical | Smoking status (`yes`, `no`) |
| `region` | Categorical | US geographic region |
| `charges` | Continuous | Individual medical costs billed by insurance |

## Pipeline Execution Architecture

### Requirements
* Python 3.8+
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`

### Installation
```bash
git clone [https://github.com/your-username/health-insurance-svr.git](https://github.com/your-username/health-insurance-svr.git)
cd health-insurance-svr
pip install -r requirements.txt
python svr_insurance_prediction.py
