import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("number-of-births-per-year.csv")

# Handle missing values
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].replace(0, pd.NA)  # Replace zeros with NaN in numeric columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())  # Replace NaNs with the mean of each column

# Convert target variable to categorical if it should be
data['Births - Sex: all - Age: all - Variant: medium'] = pd.cut(data['Births - Sex: all - Age: all - Variant: medium'], bins=3, labels=['Low', 'Medium', 'High'])

# Encode categorical columns
label_encoders = {}
for column in ['Entity', 'Code']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))  # Convert to string to avoid conversion issues
    label_encoders[column] = le

# Encode target variable
target_le = LabelEncoder()
data['Births - Sex: all - Age: all - Variant: medium'] = target_le.fit_transform(data['Births - Sex: all - Age: all - Variant: medium'])

# Define features and target variable
X = data[['Entity', 'Code', 'Year', 'Births - Sex: all - Age: all - Variant: estimates']]
y = data['Births - Sex: all - Age: all - Variant: medium']

# Drop rows with missing target values, if any
X = X[y.notna()]
y = y.dropna()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the KNN Classifier
knn_clf = KNeighborsClassifier(n_neighbors=20)  # You can change the value of k (number of neighbors)
knn_clf.fit(X_train, y_train)

# Make predictions
y_pred = knn_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print Classification Report
report = classification_report(y_test, y_pred, target_names=target_le.classes_, zero_division=1, output_dict=True)

print("\nClassification Report:\n")
print(f"precision\trecall\tf1-score\tsupport")
for key in target_le.classes_:
    if key in report:
        print(f"{key}\t{report[key]['precision']:.2f}\t{report[key]['recall']:.2f}\t{report[key]['f1-score']:.2f}\t{int(report[key]['support'])}")
    else:
        print(f"{key}\t0.00\t0.00\t0.00\t0")

print(f"\naccuracy\t\t\t{report['accuracy']:.2f}")
print(f"macro avg\t{report['macro avg']['precision']:.2f}\t{report['macro avg']['recall']:.2f}\t{report['macro avg']['f1-score']:.2f}\t{int(report['macro avg']['support'])}")
