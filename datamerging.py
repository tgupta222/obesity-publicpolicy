# import statements
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict


# loading datasets
acs_data = pd.read_csv(r'C:/Users/tanis/Documents/CDS 492/Datasets/ACSST1Y2023.S1501-Data.csv')
usda_data = pd.read_excel(r'C:/Users/tanis/Documents/CDS 492/Datasets/USDAfoodaccess.xlsx', engine='openpyxl')
bmi_data = pd.read_csv(r'C:/Users/tanis/Documents/CDS 492/Datasets/LakeCounty_Health_2397514566901885190.csv')

# # Preview the datasets
# print(acs_data.head())
# print(usda_data.head())
# print(bmi_data.head())

# DATA CLEANING

acs_data.dropna(axis=1, inplace=True) # dropping columns with NULL values
acs_data.drop(index=0).reset_index(drop=True) # dropping first row
acs_data = acs_data.loc[:, ~(acs_data.isin(['(X)'])).any()]
#print(acs_data.head())
#print(acs_data.columns)

usda_data.dropna(axis=1, inplace=True) # dropping columns with NULL values
#print(usda_data.columns)
usda_data.drop('County', axis=1, inplace=True) # aggregating by state so do not need county column

# Aggregating USDA data to be by state : to be able to merge with the other 2 datasets
aggregated_data = usda_data.groupby('State').agg(
    # Sum for numeric columns
    Pop2010=('Pop2010', 'sum'),
    OHU2010=('OHU2010', 'sum'), # occupied housing units
    
    # Mean for binary columns
    Urban=('Urban', 'mean'), 
    GroupQuartersFlag=('GroupQuartersFlag', 'mean'),
    LILATracts_1And10=('LILATracts_1And10', 'mean'),
    LILATracts_halfAnd10=('LILATracts_halfAnd10', 'mean'),
    LILATracts_1And20=('LILATracts_1And20', 'mean'),
    LILATracts_Vehicle=('LILATracts_Vehicle', 'mean'),
    HUNVFlag=('HUNVFlag', 'mean'),
    LowIncomeTracts=('LowIncomeTracts', 'mean'),
    LA1and10=('LA1and10', 'mean'),
    LAhalfand10=('LAhalfand10', 'mean'),
    LA1and20=('LA1and20', 'mean'),
    LATracts_half=('LATracts_half', 'mean'),
    LATracts1=('LATracts1', 'mean'),
    LATracts10=('LATracts10', 'mean'),
    LATracts20=('LATracts20', 'mean'),
    LATractsVehicle_20=('LATractsVehicle_20', 'mean')
).reset_index()

aggregated_data.to_excel("USDAclean.xlsx", index=False)
#print(aggregated_data)
#print(aggregated_data.head())
#print(acs_data.head())
#print(bmi_data.head())

# MERGING DATASETS

aggregated_data.rename(columns={'State': 'NAME'}, inplace=True) # Changing column name to be consistent with other datasets
merge_data = pd.merge(aggregated_data, acs_data, on='NAME', how='inner') #merging based on state column 
merged_data = pd.merge(merge_data, bmi_data, on='NAME', how='inner')

merged_data['Obese'] = (merged_data['Obesity'] >= 30).astype(int)

# preview merged data
# print(merged_data.head())
# print(merged_data.columns)

# EXPLORATORY DATA ANALYSIS

# performance metrics AUC-ROC, Accuracy, F1 score
# SVM MODEL

# KNN MODEL

# Split features and target
X = merged_data.drop(columns=['Obese', 'NAME'])  # Adjust 'NAME' if it is the identifier column
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1)
y = merged_data['Obese']

# print("Columns retained for modeling:")
# print(X.columns)

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Baseline KNN Model with 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_baseline = KNeighborsClassifier(n_neighbors=5) 
knn_baseline.fit(X_train, y_train)
y_pred_baseline = knn_baseline.predict(X_test)

# Baseline performance metrics
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_auc_roc = roc_auc_score(y_test, knn_baseline.predict_proba(X_test)[:, 1])

print("Baseline Model Performance:")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall: {baseline_recall:.4f}")
print(f"AUC-ROC: {baseline_auc_roc:.4f}")

# Grid Search with Cross-Validation
param_grid = {
    'n_neighbors': range(1, 21),  # Test k from 1 to 20
    'weights': ['uniform', 'distance'],  # Weighting schemes
}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best parameters and score from Grid Search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# print("\nGrid Search Results:")
# print(f"Best Parameters: {best_params}")
# print(f"Best Cross-Validated Accuracy: {best_score:.4f}")

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_neighbors': range(1, 21),  # Test k from 1 to 20
    'weights': ['uniform', 'distance'],  # Weighting schemes
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Extract the best parameters and create the optimized KNN model
best_params = grid_search.best_params_
knn_optimized = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights=best_params['weights'])

# Perform cross-validation with the optimized model
cv_accuracy = cross_val_score(knn_optimized, X, y, cv=5, scoring='accuracy').mean()
cv_precision = cross_val_score(knn_optimized, X, y, cv=5, scoring='precision').mean()
cv_recall = cross_val_score(knn_optimized, X, y, cv=5, scoring='recall').mean()

# Predict probabilities for AUC-ROC
y_pred_proba_cv = cross_val_predict(knn_optimized, X, y, cv=5, method='predict_proba')
cv_auc_roc = roc_auc_score(y, y_pred_proba_cv[:, 1])

# Train on the full training set and evaluate on the test set
knn_optimized.fit(X_train, y_train)
y_pred_test = knn_optimized.predict(X_test)
y_pred_test_proba = knn_optimized.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_auc_roc = roc_auc_score(y_test, y_pred_test_proba)

# Print the best parameters and cross-validation metrics
print("Optimized KNN Grid Search Results:")
print(f"Best Parameters: {best_params}")
print("\nCross-Validation Model Performance:")
print(f"Accuracy: {cv_accuracy:.4f}")
print(f"Precision: {cv_precision:.4f}")
print(f"Recall: {cv_recall:.4f}")
print(f"AUC-ROC: {cv_auc_roc:.4f}")

# Print test set performance
print("\nTest Set Performance (After Grid Search):")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"AUC-ROC: {test_auc_roc:.4f}")