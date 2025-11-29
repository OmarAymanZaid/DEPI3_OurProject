# 🧠 Stroke Risk Prediction: Proactive Clinical Decision Support

This project develops a complete end-to-end machine learning pipeline to predict the **probability of stroke** in patients using clinical, demographic, and lifestyle features. The goal is to create a deployable model for proactive stroke detection and clinical decision support.

It includes data preprocessing, exploratory analysis, model development with class balancing, evaluation, and a full deployment setup using **FastAPI** and **Docker**.

**Dataset Source:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
**Documentation:** (https://drive.google.com/drive/folders/1wZMJ2eGYaO-pen2NjA-mmrJ3Oz3z3ODy?usp=sharing)

---

## 📋 Table of Contents

1.  [🌐 Project Overview](#-project-overview)
2.  [🔧 Data Pipeline](#-data-pipeline)
3.  [🔍 Data Insights](#-data-insights)
4.  [🤖 Model Development](#-model-development)
5.  [🚀 Deployment Workflow](#-deployment-workflow)
6.  [🧩 Challenges](#-challenges)
7.  [🔑 Key Insights](#-key-insights)
8.  [🏥 Workflow Integration Recommendations](#-workflow-integration-recommendations)
9.  [🏁 Conclusion](#-conclusion)

---

## 🌐 Project Overview

The core objective is to predict stroke risk based on patient data and provide a deployable model that can integrate seamlessly into existing clinical workflows. Since stroke is a leading cause of death worldwide, early risk identification can significantly improve patient outcomes.

### Project Deliverables:

* A **validated predictive model** with a high-priority recall score.
* A complete, version-controlled machine learning pipeline.
* A **FastAPI-based** prediction service for real-time inference.
* A reliable, reproducible, **containerized deployment** setup using Docker.

---

## 🔧 Data Pipeline

### 1. Data Collection & Preprocessing

| Step | Detail |
| :--- | :--- |
| **Dataset Size** | 5,110 records, 12 features |
| **Missing Values** | **BMI** column imputed using the mean. |
| **Outliers** | Capped using the **Interquartile Range (IQR)** method for `avg_glucose_level` and `bmi`. |
| **Feature Scaling** | **Z-score standardization** applied to all numeric features. |
| **Encoding** | Performed later during model training. |

---

## 🔍 Data Insights

### Top Risk Factors

* **Age** was identified as the **strongest predictor**.
* **Heart disease**, **hypertension**, and **average glucose level** showed strong correlations with stroke risk.
* Stroke rates increased sharply in individuals **55+ years**.

### Lifestyle Patterns

* **Former smokers** showed the highest stroke prevalence (**7.91%**).
* **Self-employed** individuals had the highest risk within occupation groups.
* **Married** individuals showed significantly higher stroke rates.

> These findings align with known medical literature and provide valuable lifestyle-based public health insights.

---

## 🤖 Model Development

### Models Evaluated

* **Logistic Regression**
* **Random Forest Classifier**

> Both models were trained with **class-weight balancing** due to the severe target imbalance (stroke cases are rare).

### Optimization

| Model | Optimization Technique |
| :--- | :--- |
| **Logistic Regression** | Grid Search Cross-Validation |
| **Random Forest** | Randomized Search Cross-Validation |

### Final Model Selection: Tuned Logistic Regression

The **Tuned Logistic Regression** model was chosen based on the following performance metrics and clinical considerations:

| Metric | Score | Clinical Rationale |
| :--- | :--- | :--- |
| **Recall** | **0.84** | **Critical for identifying true stroke cases (minimizing false negatives).** |
| **ROC-AUC** | **0.841** | Strong overall risk discrimination ability. |
| **Precision & F1** | Balanced | Good trade-off between identifying true positives and overall accuracy. |
| **Interpretability** | High | Essential for transparency and adoption in clinical use. |

> **High Recall** ensures fewer **False Negatives** (missed stroke cases), which is essential in life-threatening medical predictions.

---

## 🚀 Deployment Workflow

The project utilizes a modern, maintainable development and deployment environment:

| Tool | Purpose |
| :--- | :--- |
| **Poetry** | Dependency management and virtual environment creation. |
| **Ruff** | Code linting and formatting standards enforcement. |
| **Git + Git LFS** | Version control for code and handling large model files. |
| **FastAPI** | Serving the model as a fast, asynchronous REST API. |
| **Docker** | Creating reproducible, isolated, and containerized deployments. |
| **Makefile** | Automating common tasks (`run`, `build`, `test`, etc.). |

### Deployment Options

The system is designed for flexible integration:

* As a standalone **FastAPI service**.
* As a portable **Docker container**.
* Integrated into visualization dashboards (e.g., Streamlit).
* Connected directly to **Electronic Health Record (EHR)** systems.

---

## 🧩 Challenges

### 1. Target Imbalance
* **Problem:** Stroke cases were rare in the dataset (minority class).
* **Solution:** Handled effectively using **class weighting** to prevent model bias towards the majority (non-stroke) class.

### 2. Missing BMI Values
* **Problem:** The `bmi` column had missing values.
* **Solution:** Addressed with **mean imputation**, mindful of the Missing At Random (MAR) assumptions.

### 3. Outliers
* **Problem:** Outliers in features like `avg_glucose_level` initially skewed statistical correlations.
* **Solution:** Fixed with the **IQR capping** method during the preprocessing phase.

---

## 🔑 Key Insights

### 1. Age & Cardiovascular Health Are Dominant Predictors
The strongest predictive features are **Age**, **Hypertension**, **Heart Disease**, and **Glucose Levels**, reinforcing known medical risk factors.

### 2. Recall Matters Most in Healthcare
Given the high cost of a **False Negative** (missing a high-risk patient), the model was optimized to prioritize **high recall (0.84)** over precision.

### 3. Lifestyle Contributions
Smoking history and occupation categories showed notable risk patterns, demonstrating the model's relevance to **public health interventions**.

---

## 🏥 Workflow Integration Recommendations

### 1. Clinical Decision Support (CDS)
* **Mechanism:** Deploy the model as a **REST API** connected to EHR systems.
* **Action:** During a routine checkup, the EHR can send patient features to the API, which instantly returns a **Stroke Risk Score**.

### 2. Triage & Prioritization
* **Mechanism:** Define a critical risk threshold (e.g., probability > X%).
* **Action:** High-risk patients should be:
    * **Flagged automatically** within the EHR.
    * **Prioritized** for specialist referral.
    * Given tailored monitoring or interventions.

### 3. Continuous Monitoring
* **Mechanism:** Implement a feedback loop between clinical outcomes and the prediction service.
* **Action:** Regularly **track outcomes**, **revalidate predictions**, and **retrain or recalibrate** the model over time to ensure sustained predictive reliability.

---

## 🏁 Conclusion

This project successfully delivered a full machine learning solution for stroke prediction, spanning data cleaning, comprehensive exploration, model tuning, and robust deployment.

### Project Highlights:

* Final Model: **Tuned Logistic Regression**.
* **Recall: 0.84** (Strong sensitivity to high-risk cases).
* **ROC-AUC: 0.841** (Reliable risk discrimination).
* **Fully deployable** via FastAPI and Docker.

The final model provides a robust, interpretable, and clinically meaningful foundation for proactive stroke prevention and is ready for immediate integration into clinical decision support workflows.
