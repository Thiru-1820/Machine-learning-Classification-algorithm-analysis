import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("number-of-births-per-year.csv")

# Handle missing values
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].replace(0, pd.NA)  # Replace zeros with NaN in numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Replace NaNs with the mean of each column

# Encode categorical columns
label_encoders = {}
for column in ['Entity', 'Code']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string to avoid conversion issues
    label_encoders[column] = le

# Define features and target variable
target_column = 'Births - Sex: all - Age: all - Variant: medium'

# Encode target variable
target_le = LabelEncoder()
data[target_column] = target_le.fit_transform(data[target_column])

# Define features and target variable
X = data[['Entity', 'Code', 'Year', 'Births - Sex: all - Age: all - Variant: estimates']]
y = data[target_column]

# Drop rows with missing target values, if any
X = X[y.notna()]
y = y.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print Classification Report
report = classification_report(y_test, y_pred, labels=target_le.transform(target_le.classes_), zero_division=1, output_dict=True)
print("\nClassification Report:\n", report)

# Plot Feature Importance as a Bar Chart
plt.figure(figsize=(10, 6))
feature_importances = clf.feature_importances_
features = X.columns

# Creating a DataFrame for the feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Bar plot of feature importance
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue', edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
plt.tight_layout()
plt.show()


