import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
label_encoder = LabelEncoder()
imputer = SimpleImputer(strategy='mean')

import matplotlib.pyplot as plt
import seaborn as sns 

# Read CSV File
data = pd.read_csv("titanic.csv")
# Info on the dataset
data.info()  # Display dataset information
print(data.isnull().sum())  # Check for missing values

# Data Cleaning and Feature Engineering 
def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)  # Drop irrelevant columns

    
    
    # Apply label encoding to the 'Embarked' column
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    df["Embarked"] = df["Embarked"].fillna("S")  # Fill missing 'Embarked' values with "S"

    # Convert Gender to numeric values (1 = male, 0 = female)
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})

    # Feature Engineering: Create new features
    df["FamilySize"] = df["SibSp"] + df["Parch"]  # Total family size
    df["isAlone"] = np.where(df["FamilySize"] == 0, 1, 0)  # Is the person alone (1 if yes, 0 if no)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)  # Bin Fare into 4 categories
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)  # Bin Age into 5 categories

    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Apply imputation to numeric columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

# Fill in missing ages based on Pclass median
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()  # Use median of Pclass for missing ages
    
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

# Update data
data = preprocess_data(data)
fill_missing_ages(data)  # Fill missing ages

# Create Features/Target Variables
x = data.drop(columns=["Survived"])  # Features (excluding 'Survived')
y = data["Survived"]  # Target variable (Survived)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy="mean")  # Use mean imputation for missing values
x_train = imputer.fit_transform(x_train)  # Apply imputation to training data
x_test = imputer.transform(x_test)  # Apply imputation to test data

# ML Preprocessing: Scaling features
scaler = MinMaxScaler()  # Rescale features to [0, 1]
x_train = scaler.fit_transform(x_train)  # Fit and transform training data
x_test = scaler.transform(x_test)  # Transform test data

# Hyperparameter Tuning - KNN
def tune_model(x_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),  # Trying n_neighbors from 1 to 20
        "metric": ["euclidean", "manhattan", "minkowski"],  # Distance metrics
        "weights": ["uniform", "distance"]  # Weights options (uniform or distance-based)
    }

    model = KNeighborsClassifier()  # Initialize KNN model
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)  # Grid search with cross-validation
    grid_search.fit(x_train, y_train)  # Fit model to data
    return grid_search.best_estimator_  # Return the best model

best_model = tune_model(x_train, y_train)  # Find the best KNN model

# Predictions and evaluate
def evaluate_model(model, x_test, y_test):
    prediction = model.predict(x_test)  # Predict on the test set
    accuracy = accuracy_score(y_test, prediction)  # Calculate accuracy
    matrix = confusion_matrix(y_test, prediction)  # Calculate confusion matrix
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test, y_test)  # Evaluate the best model

# Output results
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print accuracy
print(f'Confusion Matrix: \n{matrix}')  # Print confusion matrix


# Plot or Visualise
def plot_model(matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()

plot_model(matrix)