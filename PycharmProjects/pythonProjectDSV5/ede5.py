import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Generate synthetic dataset for zoo animals
np.random.seed(0)

# Generate synthetic data
n_samples = 1000
hair = np.random.randint(0, 2, size=n_samples)
feathers = np.random.randint(0, 2, size=n_samples)
eggs = np.random.randint(0, 2, size=n_samples)
milk = np.random.randint(0, 2, size=n_samples)
airborne = np.random.randint(0, 2, size=n_samples)
aquatic = np.random.randint(0, 2, size=n_samples)
predator = np.random.randint(0, 2, size=n_samples)
toothed = np.random.randint(0, 2, size=n_samples)
backbone = np.random.randint(0, 2, size=n_samples)
breathes = np.random.randint(0, 2, size=n_samples)
venomous = np.random.randint(0, 2, size=n_samples)
fins = np.random.randint(0, 2, size=n_samples)
legs = np.random.randint(0, 6, size=n_samples)  # Number of legs can be 0-5
tail = np.random.randint(0, 2, size=n_samples)
domestic = np.random.randint(0, 2, size=n_samples)
catsize = np.random.randint(0, 2, size=n_samples)
class_type = np.random.randint(1, 8, size=n_samples)  # Class type ranges from 1 to 7

# Create DataFrame
df = pd.DataFrame({
    'hair': hair,
    'feathers': feathers,
    'eggs': eggs,
    'milk': milk,
    'airborne': airborne,
    'aquatic': aquatic,
    'predator': predator,
    'toothed': toothed,
    'backbone': backbone,
    'breathes': breathes,
    'venomous': venomous,
    'fins': fins,
    'legs': legs,
    'tail': tail,
    'domestic': domestic,
    'catsize': catsize,
    'class_type': class_type
})

# Step 2: Split data into features (X) and target (y)
X = df.drop('class_type', axis=1)
y = df['class_type']

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Generate confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
