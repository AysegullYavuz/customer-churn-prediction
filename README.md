# End-to-End Enterprise Customer Churn Analysis & Prediction

This project is an end-to-end computer engineering portfolio project developed to analyze customer churn for a telecommunications company. It encompasses database-level data cleansing, interactive business intelligence dashboard design, and machine learning pipeline implementation to predict future customer attrition.

## 🚀 Project Architecture & Core Competencies
The project is built upon 3 primary engineering layers:
1. **Data Engineering & SQL:** Ingesting raw data into SQL Server, optimizing data types, handling missing/corrupted values, and building dynamic, analysis-ready SQL Views.
2. **Business Intelligence (BI) & Analytics:** Transforming cleansed data into interactive, filter-driven executive dashboards using Power BI and Tableau to support strategic decision-making.
3. **Data Science & Machine Learning:** Performing feature engineering, handling class imbalance, and training high-performance predictive models using Python (`pandas`, `numpy`, `scikit-learn`, `xgboost`).

---

## 📊 1. Database Layer (SQL Server)
The raw dataset was ingested into a relational database and processed using the scripts located in the `sql/` directory:
* Fixed improperly formatted data types (e.g., converting `TotalCharges` to appropriate numeric formats).
* Resolved logical inconsistencies and deployed optimized dynamic analytics views (`v_CleanedChurnData`) for seamless BI integration.

---

## 📉 2. Business Intelligence & Executive Dashboards (Power BI & Tableau)
To enable stakeholders to extract actionable strategic insights, the data was modeled across two major BI platforms:

### 🔹 Power BI Dashboard
* Modeled dynamic churn distributions categorized by contract types, internet service providers, and billing methods.
* The source file is available under the `dashboards/` directory.

### 🔹 Tableau Public Dashboard
* A live, interactive executive dashboard deployed on the web with optimized cross-filtering capabilities.
* **Live Project Link:** [Explore the Interactive Dashboard on Tableau Public]([https://public.tableau.com/app/profile/ay.eg.l.yavuz/viz/Enterprise_Customer_Churn_Tableau/Dashboard1#1])

<img width="830" height="776" alt="tableau_screenshot" src="https://github.com/user-attachments/assets/74f2cdd8-be78-49a5-85ec-8ac236f8d9b3" />

---

## 🤖 3. Machine Learning Model (Python)
A robust machine learning pipeline was constructed in Python to proactively detect customers who are at high risk of churning:
* **Stack:** pandas, numpy, scikit-learn, xgboost
* **Feature Engineering:** Implemented categorical encoding, missing data imputation, and feature scaling.
* **Modeling:** Trained advanced classification algorithms and optimized model performance utilizing evaluation metrics specific to highly imbalanced target variables.
