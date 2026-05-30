#J48
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Encode categorical attributes
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# J48 equivalent Decision Tree
model = DecisionTreeClassifier(criterion='entropy')

# Cross-validation
scores = cross_val_score(model, X, y, cv=10)

# Train model
model.fit(X, y)

# Print Decision Tree
tree_rules = export_text(model, feature_names=list(X.columns))
print("J48 Decision Tree:\n")
print(tree_rules)

print("\nCross Validation Accuracy:")
print(scores)
print("\nAverage Accuracy:")
print(scores.mean() * 100)


#PART
# Train shallow tree to generate readable rules
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Cross-validation
scores = cross_val_score(model, X, y, cv=10)

# Train model
model.fit(X, y)

# Extract rules
rules = export_text(model, feature_names=list(X.columns))
print("PART-like Rule Set:\n")
print(rules)

print("\nAverage Accuracy:")
print(scores.mean() * 100)



#OneR
# Select only one important attribute (e.g., credit_history)
X = data[['credit_history']]
y = data.iloc[:, -1]

# Decision stump = OneR style classifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=1)

# Cross-validation
scores = cross_val_score(model, X, y, cv=10)

# Train model
model.fit(X, y)

# Print OneR rule
rules = export_text(model, feature_names=['credit_history'])
print("OneR Rule:\n")
print(rules)

print("\nAverage Accuracy:")
print(scores.mean() * 100)
