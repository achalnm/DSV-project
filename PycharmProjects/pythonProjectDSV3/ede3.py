import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Generate synthetic dataset and save to IrisTest_TrainData.csv
np.random.seed(0)

# Create features (sepal length, sepal width, petal length, petal width)
sepal_length = np.random.normal(5.8, 0.8, 100)
sepal_width = np.random.normal(3.1, 0.5, 100)
petal_length = np.random.normal(3.7, 0.9, 100)
petal_width = np.random.normal(1.2, 0.4, 100)

# Create target variable (species)
species = np.random.choice(['Setosa', 'Versicolor', 'Virginica'], size=100)

# Create a DataFrame
data = pd.DataFrame({
    'SepalLength': sepal_length,
    'SepalWidth': sepal_width,
    'PetalLength': petal_length,
    'PetalWidth': petal_width,
    'Species': species
})

# Save to CSV file
data.to_csv('IrisTest_TrainData.csv', index=False)

print("Dataset saved as IrisTest_TrainData.csv")

# Step 2: Load Train_Data and Test_Data from CSV files
Train_Data = pd.read_csv('IrisTest_TrainData.csv')

# Create Test_Data by splitting Train_Data (example splitting into 20% test data)
Test_Data = Train_Data.sample(frac=0.2, random_state=1)
Train_Data = Train_Data.drop(Test_Data.index)

# Save Test_Data to IrisTest_TestData.csv
Test_Data.to_csv('IrisTest_TestData.csv', index=False)

print("Dataset split into Train and Test datasets.")

# Step 3: Answer the questions

# Question 1: Number of missing values in Train_Data
missing_values_train = Train_Data.isnull().sum().sum()
print(f"Number of missing values in Train_Data: {missing_values_train}")

# Question 2: Proportion of Setosa types in Test_Data
proportion_setosa_test = (Test_Data['Species'] == 'Setosa').mean()
print(f"Proportion of Setosa types in Test_Data: {proportion_setosa_test}")

# Question 3: Accuracy score of K-Nearest Neighbor Model (model_1)
X_train = Train_Data.drop('Species', axis=1)
y_train = Train_Data['Species']

X_test = Test_Data.drop('Species', axis=1)
y_test = Test_Data['Species']

# Create and fit K-Nearest Neighbor model with 2/3 neighbors
model_1 = KNeighborsClassifier(n_neighbors=int(2/3 * len(X_train)))
model_1.fit(X_train, y_train)

# Predict on test data and calculate accuracy
y_pred = model_1.predict(X_test)
accuracy_model_1 = accuracy_score(y_test, y_pred)
print(f"Accuracy score of model_1: {accuracy_model_1}")

# Question 4: Indices of Misclassified Samples from model_1
misclassified_indices = y_test.index[y_pred != y_test]
print(f"Indices of misclassified samples from model_1: {misclassified_indices}")

# Question 5: Accuracy of Logistic Regression Model (model_2)
# Create and fit Logistic Regression model
model_2 = LogisticRegression()
model_2.fit(X_train, y_train)

# Predict on test data and calculate accuracy
y_pred_model_2 = model_2.predict(X_test)
accuracy_model_2 = accuracy_score(y_test, y_pred_model_2)
print(f"Accuracy score of model_2 (Logistic Regression): {accuracy_model_2}")
