from DataPreperation import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve


# KNN MODEL

# Split features and target
X = merged_data.drop(columns=['Obese', 'NAME'])  # Adjust 'NAME' if it is the identifier column
X = X.select_dtypes(include=[np.number])
X = X.dropna(axis=1)
y = merged_data['Obese']

# print("Columns retained for modeling:")
# print(X.columns)

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