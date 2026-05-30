import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv(r"C:\Users\PC\GermanCredit.csv")

# Encode categorical attributes
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == object:
        data[column] = label_encoder.fit_transform(data[column])

# Features and target class
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# -----------------------------
# SVM Classifier
# -----------------------------
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy * 100)
print("\nSVM Confusion Matrix:\n", confusion_matrix(y_test, svm_predictions))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_predictions))

# -----------------------------
# Decision Tree Classifier
# -----------------------------
dt_model = DecisionTreeClassifier(criterion='entropy')
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)
print("\nDecision Tree Accuracy:", dt_accuracy * 100)
print("\nDecision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, dt_predictions))
