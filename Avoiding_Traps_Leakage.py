from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

#Task 1 — Reproduce and Identify Leakage

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

#Scale & split data
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_predic_train = model.predict(X_train)
y_predic_test = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_predic_train)
test_acc = accuracy_score(y_test, y_predic_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

#Task 2 - Fix the workflow using a pipeline

# combines StandardScaler and LogisticRegression.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Split the data first using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run 5-fold cross-validation
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)

# Report mean accuracy and standard deviation
print(f"Mean CV Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")

#Task 3 — Experiment with Decision Tree Depth

# decision tree with different max_depth values
max_depths = [1, 5, 20]

#same train-test split from Task 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Record train and test accuracy for each depth

for depth in max_depths:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, tree_model.predict(X_train))
    test_acc = accuracy_score(y_test, tree_model.predict(X_test))
    
    print(f"Max Depth: {depth} | Training Accuracy: {train_acc:.2f} | Testing Accuracy: {test_acc:.2f}")

# Max Depth 5 best balances fit and generalization.

#1. Max Depth 1: Underfitting, low training and testing accuracy.
#2. Max Depth 5: Good balance, high training and testing accuracy.
#3. Max Depth 20: Overfitting, high training accuracy but low testing accuracy.
