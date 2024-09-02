import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Step 1: Load the dataset
data = pd.read_csv('./number-of-births-per-year.csv')  # Update this with the correct path to your dataset

# Step 2: Preprocess the data
# Dropping non-relevant columns
X = data.drop(columns=['Entity', 'Code', 'Year', 'Births - Sex: all - Age: all - Variant: medium'])  # Exclude target variable

# Creating a binary target variable (adjust this based on your dataset)
y = (data['Births - Sex: all - Age: all - Variant: medium'] > data['Births - Sex: all - Age: all - Variant: medium'].median()).astype(int)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_clean = imputer.fit_transform(X)

# Step 3: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Step 4: Apply PCA if you have more than one feature
if X_scaled.shape[1] > 1:
    pca = PCA(n_components=2)  # Reduce to 2 principal components
    X_pca = pca.fit_transform(X_scaled)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
else:
    X_pca = X_scaled
    print("Skipping PCA because there's only one feature.")

# Display the first 15 rows of the PCA-transformed dataset
print("PCA-transformed data (first 15 rows):\n", X_pca[:15])

# Step 5: Split the PCA data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 6: Train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = classifier.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the first 15 actual and predicted values for comparison
print("\nFirst 15 Actual vs Predicted Values:")
for actual, predicted in zip(y_test[:15], y_pred[:15]):
    print(f"Actual: {actual}, Predicted: {predicted}")

