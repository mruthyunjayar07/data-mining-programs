# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Display first few rows
print("Dataset Sample:")
print(data.head())

# Convert categorical attributes into numeric values
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Separate features and target class
X = data.iloc[:, :-1]   # all columns except last
y = data.iloc[:, -1]    # last column as target

# Train Decision Tree using complete dataset
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)
import matplotlib.pyplot as plt
from sklearn import tree

# Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in set(y)], filled=True)
plt.show()