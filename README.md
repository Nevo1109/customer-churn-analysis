# Customer Churn Analysis

## Overview

This project analyzes customer churn behavior using machine learning techniques.  
The goal is to identify key factors that influence customer retention and build predictive models to estimate churn probability.

The analysis includes data preprocessing, exploratory data analysis, model training, and feature importance evaluation.

---

## Business Problem

Customer churn is a critical issue in subscription-based businesses.  
Understanding why customers leave helps companies improve retention strategies and reduce revenue loss.

---

## Dataset

The dataset contains telecom customer data including:

- Customer demographics
- Subscription details
- Contract type
- Payment method
- Service usage information

---

## Workflow

1. Data Loading  
2. Data Cleaning (handling missing values, encoding categorical variables)  
3. Exploratory Data Analysis (EDA)  
4. Model Training  
5. Model Evaluation  
6. Feature Importance Analysis  

---

## Machine Learning Models

The following models were used:

- Logistic Regression  
- Random Forest Classifier  

---

## Results

The Random Forest model performed better than Logistic Regression.

### Key Metrics:
- Accuracy: ~78%

### Important Features:
- Total Charges
- Monthly Charges
- Tenure
- Contract Type

---

## Key Insights

- Customers with shorter tenure are more likely to churn  
- Higher monthly charges increase churn probability  
- Long-term contracts significantly reduce churn risk  

---

## Technologies Used

- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

---

## How to Run the Project

```bash
git clone https://github.com/Nevo1109/customer-churn-analysis.git
cd customer-churn-analysis
pip install -r requirements.txt
python main.py
