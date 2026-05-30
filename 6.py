import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Convert categorical attributes into numeric values
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Select attributes by index (2, 3, 5, 7, 10, 17)
# Python uses 0-based indexing, so attribute 2 = index 1, etc.
selected_indices = [1, 2, 4, 6, 9, 16]   # duration, credit_history, amount, employment_duration, other_debtors, job
selected_columns = data.columns[selected_indices]
X = data[selected_columns]

# Target class attribute (attribute 21 → index 20)
y = data[data.columns[20]]  # credit_risk

# Create Decision Tree classifier
model = DecisionTreeClassifier(criterion='entropy')

# Fit the model
model.fit(X, y)

# Predict on the same dataset (training set)
y_pred = model.predict(X)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred) * 100
print("Training Accuracy using predict():", accuracy)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in set(y)], filled=True)
plt.show()