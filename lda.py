import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE  # Import SMOTE from imbalanced-learn

# Step 1: Load the dataset
data = pd.read_csv('./number-of-births-per-year.csv')  # Update this with the correct path to your dataset

# Step 2: Preprocess the data
# Drop non-relevant columns and set the target variable
X = data.drop(columns=['Entity', 'Code', 'Year'])  # Adjust as necessary
y = data['Births - Sex: all - Age: all - Variant: medium']  # Replace with actual target column name

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_clean = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Step 3: Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_scaled, y)

# Apply LDA
lda = LDA(n_components=1)  # Reduce to 1 linear discriminant component
X_lda = lda.fit_transform(X_smote, y_smote)

# Display the LDA-transformed data
print("Explained variance ratio (Not applicable for LDA): N/A")  # LDA does not provide variance ratio like PCA
print("LDA-transformed data (first 5 rows):\n", X_lda[:5])

# Step 4: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y_smote, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after LDA:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

