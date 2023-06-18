# Home Credit Default Risk Prediction

This project focuses on predicting each applicant's ability to repay the loan.

# File Explanation on Repository

**deployment** : Contains files used for deployment to HuggingFace

**Final_Project_Novanska_Aginta.ipynb** : notebooks related to modeling, EDA , prediction for this project

**url.txt** link to my huggingface deployment

# Summary of This Project

Stages in the project is: 

- First,  EDA to determine the distribution of each column that has a relationship to the data

- Second, cleaning,preprocessing the data such as impute the missing values, scaling etc.

- Third is modeling, i use metrics AUC for this modeling because auc have ability to distinguish between different classes. The higher the AUC value, the better the model's ability to distinguish between different classes.

with the results :
1. AUC Logistic Regression: 0.7335502751814258
2. AUC Random Forest: 0.7335502751814258
3. AUC XGBoost: 0.7395828210184866

# Project Conclussion
From 3 models we know that :
- When viewed from the classification report on y_test from all models it is still seen that this model is still too biased towards 0 and there are still many misclassifications on 1
- The auc score is not much different as u can see at summary of this project, but XGBoost have better AUC Score than 2 models
- However, if you look at the classification report on y_train, this model is very good at predicting 0 or 1 because I previously did SMOTE on y_train (balancing).
