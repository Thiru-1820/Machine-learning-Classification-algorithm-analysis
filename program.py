import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier

# Load real-time dataset (specify encoding to handle non-UTF-8 characters)
data = pd.read_csv('dataset.csv', encoding='ISO-8859-1')

# Encode categorical variables (if any)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Data preprocessing (assumes the last column is the target)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the number of features
n_features = X_scaled.shape[1]
print(f"Number of features: {n_features}")

# Feature Selection: SelectKBest (univariate feature selection) using f_classif
k = min(3, n_features)  # Set k to be the number of features available
select_k_best = SelectKBest(score_func=f_classif, k=k)
X_new = select_k_best.fit_transform(X_scaled, y)

# Feature Selection: RFE (Recursive Feature Elimination) with DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator=tree_clf, n_features_to_select=min(3, n_features))  # Set n_features_to_select to the number of features available
X_rfe = rfe.fit_transform(X_scaled, y)

# Splitting the dataset into training and testing sets (for each feature selection method)
X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_new, y, test_size=0.3, random_state=42)
X_train_rfe, X_test_rfe, y_train_rfe, y_test_rfe = train_test_split(X_rfe, y, test_size=0.3, random_state=42)

# Initialize classifiers
log_reg = LogisticRegression(random_state=42, max_iter=10000, solver='liblinear')  # Increased max_iter and used 'liblinear' solver
rf_clf = RandomForestClassifier(random_state=42)

# Train and evaluate classifiers with SelectKBest
log_reg.fit(X_train_k, y_train_k)
rf_clf.fit(X_train_k, y_train_k)
y_pred_log_k = log_reg.predict(X_test_k)
y_pred_rf_k = rf_clf.predict(X_test_k)
acc_log_k = accuracy_score(y_test_k, y_pred_log_k)
acc_rf_k = accuracy_score(y_test_k, y_pred_rf_k)

# Train and evaluate classifiers with RFE
log_reg.fit(X_train_rfe, y_train_rfe)
rf_clf.fit(X_train_rfe, y_train_rfe)
y_pred_log_rfe = log_reg.predict(X_test_rfe)
y_pred_rf_rfe = rf_clf.predict(X_test_rfe)
acc_log_rfe = accuracy_score(y_test_rfe, y_pred_log_rfe)
acc_rf_rfe = accuracy_score(y_test_rfe, y_pred_rf_rfe)

# Plot the results
labels = ['Logistic Regression (SelectKBest)', 'Random Forest (SelectKBest)',
          'Logistic Regression (RFE)', 'Random Forest (RFE)']
accuracies = [acc_log_k, acc_rf_k, acc_log_rfe, acc_rf_rfe]

plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color=['skyblue', 'orange', 'skyblue', 'orange'])
plt.xlabel('Classifiers with Feature Selection')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Algorithms with Feature Selection')
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
plt.show()

# Print results
print(f"Logistic Regression (SelectKBest) Accuracy: {acc_log_k:.2f}")
print(f"Random Forest (SelectKBest) Accuracy: {acc_rf_k:.2f}")
print(f"Logistic Regression (RFE) Accuracy: {acc_log_rfe:.2f}")
print(f"Random Forest (RFE) Accuracy: {acc_rf_rfe:.2f}")

# Determine and print the best algorithm
accuracies_dict = {
    'Logistic Regression (SelectKBest)': acc_log_k,
    'Random Forest (SelectKBest)': acc_rf_k,
    'Logistic Regression (RFE)': acc_log_rfe,
    'Random Forest (RFE)': acc_rf_rfe
}

best_algorithm = max(accuracies_dict, key=accuracies_dict.get)
best_accuracy = accuracies_dict[best_algorithm]

print(f"\nThe best algorithm is {best_algorithm} with an accuracy of {best_accuracy:.2f}.")

