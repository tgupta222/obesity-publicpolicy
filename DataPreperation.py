import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

################## DATA PREPERATION


# LOADING DATASETS

acs_data = pd.read_csv(r'C:/Users/tanis/Documents/CDS 492/Datasets/ACSST1Y2023.S1501-Data.csv')
usda_data = pd.read_excel(r'C:/Users/tanis/Documents/CDS 492/Datasets/USDAfoodaccess.xlsx', engine='openpyxl')
bmi_data = pd.read_csv(r'C:/Users/tanis/Documents/CDS 492/Datasets/LakeCounty_Health_2397514566901885190.csv')

# # Preview the datasets
# print(acs_data.head())
# print(usda_data.head())
# print(bmi_data.head())


#ACS DATASET CLEANING

# dropping the first row, irrelevant codes from dataset
acs_data.drop(index=1).reset_index(drop=True)
# dropping columns with NULL values
acs_data.dropna(axis=1, inplace=True)
# some columns are just X's, not keeping those
acs_data = acs_data.loc[:, ~(acs_data.isin(['(X)'])).any()]

# previewing clean ACS dataset
#print(acs_data.head())
#print(acs_data.columns)


#USDA DATASET CLEANING

# # seeing all columns
# print(usda_data.columns)

#dropping colums with NULL values
usda_data.dropna(axis=1, inplace=True)
# dropping county column, aggregating by state so information is irrelavant 
usda_data.drop('County', axis=1, inplace=True)

# grouping data by state to merge with the other 2 datasets whose information is by state
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

# previewing cleaned USDA dataset
#print(aggregated_data.head())


# CLEANING BMI DATASET

#dropping colums with NULL values
bmi_data.dropna(axis=1, inplace=True)

# Making obesity binary for modeling purposes
bmi_data['Obese'] = (bmi_data['Obesity'] >= 30).astype(int)

# previewing cleaned BMI dataset
#print(BMI_data.head())


# MERGING DATASETS

# the state column is labeled NAME in 2/3 datasets, changing it to NAME in third dataset
aggregated_data.rename(columns={'State': 'NAME'}, inplace=True) 
# merging based on NAME column, each row represents a singular state
merge_data = pd.merge(aggregated_data, acs_data, on='NAME', how='inner')  
merged_data = pd.merge(merge_data, bmi_data, on='NAME', how='inner')

# preview merged data
# print(merged_data.head())
# print(merged_data.columns)


##################
##################EXPLORATORY DATA ANALYSIS


# CORRELATION ANALYSIS

# making sure there are no numerical columns in my EDA
numerical_data = merged_data.select_dtypes(include=[np.number])

# calculate the correlation matrix between features 
correlation_matrix = numerical_data.corr()

# plotting a correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# finding variables mot correlated with "obese" column
if 'Obese' in numerical_data.columns:
    correlation_with_target = correlation_matrix['Obese'].sort_values(ascending=False)
    print("\nTop Features Correlated with Obesity:")
    print(correlation_with_target.head(10))

top_correlated_features = correlation_with_target.head(5)

# plotting the top correlated features
plt.figure(figsize=(10, 6))
sns.barplot(
    y=top_correlated_features.index,
    x=top_correlated_features.values
    )
plt.title("Top Features Correlated with Obesity", fontsize=16)
plt.xlabel("Correlation Coefficient", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.show()


##################
##################MODEL TESTING


# PERPARING FEATURES AND TARGET VARIABLE

# features, dropping the state column and target variable
X = merged_data.drop(columns=['Obese', 'NAME', 'Obesity'])  
# only keeping numerical features, for safety since my models are for numeric variables
X = X.select_dtypes(include=[np.number])
# target variables, obesity
y = merged_data['Obese']

# previewing features
# print(X.columns)

# 80/20 train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# KNN MODEL

#baseline KNN model
knn_baseline = KNeighborsClassifier(n_neighbors=5) 
knn_baseline.fit(X_train, y_train)
# running the model
y_pred_baseline = knn_baseline.predict(X_test)

# baseline models performance metrics
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
baseline_precision = precision_score(y_test, y_pred_baseline)
baseline_recall = recall_score(y_test, y_pred_baseline)
baseline_auc_roc = roc_auc_score(y_test, knn_baseline.predict_proba(X_test)[:, 1])

# print("Baseline KNN Model Performance:")
# print(f"Accuracy: {baseline_accuracy:.4f}")
# print(f"Precision: {baseline_precision:.4f}")
# print(f"Recall: {baseline_recall:.4f}")
# print(f"AUC-ROC: {baseline_auc_roc:.4f}")

# performing hyperparameter tuning with cross validation to improve the baseline KNN model

# defining the parameters for GridSearchCV to test
parameters = {

    # testing k-neighbors from 1 to 20
    'n_neighbors': range(1, 21), 

    # testing two different weights 
    'weights': ['uniform', 'distance'], 

}

# grid search test to find best parameters 
# 5 fold cross validation is standard
# scoring based on f1 because data is not evenly split between obese and not obese, accuracy though standard may not be appropriate here
grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, scoring='f1')
# use our data to test parameters
grid_search.fit(X, y)
# grid search results
best_parameters = grid_search.best_params_

# new and improved knn model 

# knn model with hyperparameter tuning 
knn = KNeighborsClassifier(n_neighbors=best_parameters['n_neighbors'], weights=best_parameters['weights'])
knn.fit(X_train, y_train)
# running the model
y_pred_knn = knn.predict(X_test)

#performance metrics
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_auc_roc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

print("KNN Model Performance:")
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"AUC-ROC: {knn_auc_roc:.4f}")

#SVM MODEL

# baseline SVM model
svm_baseline = SVC(probability=True, random_state=42)  
svm_baseline.fit(X_train, y_train)
# Running the model
y_pred_svm_baseline = svm_baseline.predict(X_test)

# Baseline SVM model performance metrics
svm_baseline_accuracy = accuracy_score(y_test, y_pred_svm_baseline)
svm_baseline_precision = precision_score(y_test, y_pred_svm_baseline)
svm_baseline_recall = recall_score(y_test, y_pred_svm_baseline)
svm_baseline_auc_roc = roc_auc_score(y_test, svm_baseline.predict_proba(X_test)[:, 1])

print("Baseline SVM Model Performance:")
print(f"Accuracy: {svm_baseline_accuracy:.4f}")
print(f"Precision: {svm_baseline_precision:.4f}")
print(f"Recall: {svm_baseline_recall:.4f}")
print(f"AUC-ROC: {svm_baseline_auc_roc:.4f}")

# parameters ti test
svm_parameters = {
    # I originally wanted to test multiple kernels but run time was too long
    'kernel': ['rbf'], 
    'C': [0.1, 1, 10], 
    'gamma': ['scale', 'auto'] 
}

# # Hyperparameter Tuning with Randomized Search, gridsearch had too long of a run time on my computer
random_search_svm = RandomizedSearchCV(SVC(probability=True, random_state=42), svm_parameters, cv=3, scoring='f1', n_iter=6, n_jobs=-1)
# finding best parameters just on the 
random_search_svm.fit(X, y)
best_svm_parameters = random_search_svm.best_params_

# New and improved SVM model
svm = SVC(probability=True, random_state=42, kernel=best_svm_parameters['kernel'], 
          C=best_svm_parameters['C'], gamma=best_svm_parameters['gamma'])
svm.fit(X_train, y_train)
# Running the model
y_pred_svm = svm.predict(X_test)

# Improved SVM model performance metrics
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_auc_roc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

print("Improved SVM Model Performance:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"AUC-ROC: {svm_auc_roc:.4f}")


# RANDOM FOREST MODEL

rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)

# train the model
rf.fit(X_train, y_train)

# predictions for ROC AUC curve calculations
y_pred_rf = rf.predict(X_test)

# performance metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_auc_roc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print("Random Forest Model Performance:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"AUC-ROC: {rf_auc_roc:.4f}")

# FEATURE IMPORTANCE

# extract features
feature_importances = rf.feature_importances_
# organizing features my importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# plot features by importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title('Top 10 Features by Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# printing top 5 features
print("\nTop Features by Importance:")
print(importance_df.head(5))




