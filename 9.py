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

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create pruned Decision Tree
# max_depth and min_samples_leaf act as pruning controls
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,          # limit depth of tree
    min_samples_leaf=10   # require at least 10 samples per leaf
)

# Perform 10-fold cross validation
scores = cross_val_score(model, X, y, cv=10)

# Train model on full dataset (for visualization)
model.fit(X, y)

# Print accuracy
print("Cross Validation Accuracy for each fold:")
print(scores)
print("\nAverage Accuracy:")
print(scores.mean() * 100)

# Plot the pruned decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in set(y)], filled=True)
plt.show()
