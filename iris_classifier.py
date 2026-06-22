# Iris Flower Classification
# Name: Fiza Naz 
# Date: June 2026

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# loading the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("first 5 rows of the dataset:")
print(df.head())
print("\ntotal rows and columns:", df.shape)
print("\nhow many flowers in each class:")
print(df['species'].value_counts())

# separating features and labels
X = iris.data
y = iris.target

# scaling the data so all features are on same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nscaling done")

# splitting data into training and testing
# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("training samples:", len(X_train))
print("testing samples:", len(X_test))

# trying different values of k to find the best one
errors = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    errors.append(np.mean(pred != y_test))

# plotting the error rate for each k value
plt.figure(figsize=(10, 5))
plt.plot(range(1, 31), errors, marker='o', color='blue', linestyle='--')
plt.title('finding the best k value')
plt.xlabel('k')
plt.ylabel('error rate')
plt.grid(True)
plt.savefig('elbow_curve.png')
plt.show()

# k=5 looked good from the graph so using that
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("\nmodel training done with k=5")

# checking how well the model did
print("\nclassification report:")
print(classification_report(y_test, predictions,
      target_names=iris.target_names))

score = f1_score(y_test, predictions, average='weighted')
print("f1 score:", round(score, 4))

# confusion matrix to see which flowers were correct
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            cmap='Blues')
plt.title('confusion matrix')
plt.ylabel('actual')
plt.xlabel('predicted')
plt.savefig('confusion_matrix.png')
plt.show()

# testing with a new flower measurement
new_flower = [[5.1, 3.5, 1.4, 0.2]]
new_scaled = scaler.transform(new_flower)
result = model.predict(new_scaled)
print("\ntesting with new flower:", new_flower[0])
print("predicted species:", iris.target_names[result[0]])