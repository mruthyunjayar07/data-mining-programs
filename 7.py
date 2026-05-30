import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Convert categorical attributes into numeric values
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Select only required attributes (using correct names from your dataset)
selected_columns = [
    'credit_history',
    'purpose',
    'employment_duration',
    'other_debtors',
    'housing',
    'present_residence'
]
X = data[selected_columns]

# Target class attribute
y = data['credit_risk']

# Create Decision Tree classifier
model = DecisionTreeClassifier(criterion='entropy')

# Perform 10-fold cross validation
scores = cross_val_score(model, X, y, cv=10)

# Print results
print("Accuracy for each fold:")
print(scores)
print("\nAverage Accuracy:")
print(scores.mean() * 100)

# Train the model on the full dataset (needed for plotting)
model.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in set(y)], filled=True)
plt.show()
