# Customer_Churn_Prediction_Model

## Overview
This project aims to build a customer churn prediction model for a telecommunications company. The goal is to predict whether a customer will churn (leave the service) based on various attributes such as their contract type, payment method, tenure, and monthly charges. Accurately predicting customer churn allows businesses to take proactive actions, such as offering targeted incentives to retain customers. The project uses various machine learning techniques to achieve the best prediction performance.

## Key Objectives:
1. Data Preprocessing: Prepare the dataset for machine learning by handling categorical variables, missing values, and class imbalance.
2. Feature Engineering: Create meaningful interaction features and customer segments to improve model performance.
3. Modeling: Apply tree-based machine learning models and ensemble methods to predict customer churn.
4. Optimization: Use Bayesian Optimization to fine-tune the best-performing model (XGBoost).
5. Evaluation: Evaluate models using metrics such as accuracy, precision, recall, and F1-score, and handle class imbalance using SMOTE.

## Dataset
The dataset used for this project is the Telco Customer Churn dataset, containing 7,032 customer records with 21 features. Key features include:

- Customer Demographics: Gender, SeniorCitizen, Dependents.
- Services: PhoneService, InternetService, OnlineSecurity, TechSupport.
- Billing Information: MonthlyCharges, TotalCharges, PaymentMethod.
- Target Variable: Churn (whether the customer has left the service, represented as "Yes" or "No").

## Steps and Methodology
1. Data Preprocessing
   
**One-Hot Encoding**: Converted categorical variables into dummy variables.
**Handling Missing Values**: Ensured no missing values remained in the dataset.
**Class Imbalance**: Addressed the imbalance in the churn and non-churn customers using SMOTE.

2. Feature Engineering
   
**Interaction Features**: Created interaction terms such as tenure multiplied by monthly charges to capture relationships between customer behavior and churn.

**Customer Segmentation**: Used KMeans clustering to group customers based on tenure and spending.

**Tenure Cohorts**: Created time-based cohorts for customers based on tenure to capture the impact of time spent with the service.

3. Modeling

We applied several tree-based models:

**Decision Tree**: Used as a baseline for interpretability and feature importance analysis.

**Random Forest**: An ensemble model that aggregates multiple decision trees for improved accuracy and robustness.

**AdaBoost** and **Gradient Boosting**: Boosting algorithms that improve weak learners by iteratively correcting errors.

**XGBoost**: A high-performance implementation of gradient boosting, which was fine-tuned using Bayesian Optimization for optimal results.

4. Model Optimization
   
**Bayesian Optimization**: Efficiently tuned the hyperparameters of the XGBoost model, improving its accuracy by finding the optimal parameter combinations.

5. Model Evaluation
 
**Cross-Validation**: Applied Stratified K-Fold Cross-Validation to evaluate model performance across multiple data splits.

**Classification Metrics**: Evaluated models using accuracy, precision, recall, and F1-score to assess performance, particularly on the minority class (churned customers).

**Confusion Matrix**: Visualized model performance using confusion matrices.

## Results

- The XGBoost model, after Bayesian Optimization, achieved an accuracy of around 78%, with a balanced performance across precision and recall.
- Random Forest and Gradient Boosting also provided similar levels of accuracy (77-78%) with slightly different trade-offs between precision and recall.
- AdaBoost and Decision Tree were explored but did not outperform the ensemble models.

## Technologies and Libraries
- Python: Programming language used for the entire project.
- Pandas, NumPy: For data manipulation and preprocessing.
- Scikit-learn: For modeling, evaluation, and handling techniques like SMOTE, GridSearchCV, and RandomizedSearchCV.
- XGBoost: Gradient boosting library used for high-performance predictive modeling.
- Bayesian Optimization: For hyperparameter tuning of XGBoost.
- Seaborn, Matplotlib: For visualizations (EDA and model evaluation).

## How to Run the Project
Clone the repository:
```
bash
Copy code
git clone <repo-link>
cd churn-prediction
```

Run the Jupyter Notebook: Open `churn_prediction.ipynb` to view and run the code step-by-step.

Model Training and Testing: Follow the steps in the notebook to train the models, apply hyperparameter tuning, and evaluate performance on the test set.

## Conclusion
This project demonstrated the effectiveness of tree-based models, particularly XGBoost, in predicting customer churn. With feature engineering, SMOTE, and Bayesian Optimization, the model achieved significant accuracy while maintaining a balance between precision and recall. This solution provides a strong foundation for further tuning or integration into a business pipeline aimed at reducing customer churn.
