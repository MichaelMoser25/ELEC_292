import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, ConfusionMatrixDisplay, confusion_matrix, roc_curve, \
    RocCurveDisplay, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

df = pd.read_csv('winequalityN-lab6.csv')
# Step 3: Update the 'quality' column to have binary values
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Step 4: Drop the first column of the dataset
df = df.drop(df.columns[0], axis=1)

# Assign the last column to 'labels' and the remaining columns to 'data'
labels = df.iloc[:, -1]  # Assuming 'quality' is the last column
data = df.iloc[:, :-1]   # All columns except the last one
# Now 'labels' contains the binary labels and 'data' contains the features

# Step 5: Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train and test your model
# Initialize the logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence

# Train the model on the scaled training data
model.fit(X_train_scaled, Y_train)

# Test the model on the scaled test data
Y_pred = model.predict(X_test_scaled)


df = pd.read_csv('winequalityN-lab6.csv')
df.loc[df['quality'] <= 5, 'quality'] = 0
df.loc[df['quality'] >= 6, 'quality'] = 1
data = df.iloc[:, 1:-1]
labels = df.iloc[:, -1]

# Assign 20% of the data to test the set
x_train, x_test, y_train, y_test = \
    train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0)

# Define a standard scaler to normalize inputs
scaler = StandardScaler()

# Define classifier and the pipline
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)

#training
clf.fit(x_train, y_train)

# obtain predictions and probabilities
y_pred = clf.predict(x_test)
y_clf_prob = clf.predict_proba(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate the recall of the model
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Plot ROC curve
fpr, tpr, threshold = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# Calculate AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])
print('The AUC is: ', auc)

# for i in range(len(fpr)):
#         print(f'threshold: {threshold[i]}, tpr: {tpr[i]}, fpr: {fpr[i]}')


# Calculate F1 Score True Positives (TP), False Positives (FP), and False Negatives (FN)
# F1 = (2*tpr)/(2*tpr+fpr+FN)
# Calculate the F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# Part 2: ---------------------------------------

df = pd.read_csv('winequalityN-lab6.csv')
# Step 3: Update the 'quality' column to have binary values
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Step 4: Drop the first column of the dataset
df = df.drop(df.columns[0], axis=1)

# Assign the last column to 'labels' and the remaining columns to 'data'
labels = df.iloc[:, -1]  # Assuming 'quality' is the last column
data = df.iloc[:, :-1]   # All columns except the last one
# Now 'labels' contains the binary labels and 'data' contains the features

# Step 5: Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)

StandardScaler()
LogisticRegression(max_iter=10000)
PCA(n_components=2)

# Define the steps in the pipeline
steps = [
    ('scaler', StandardScaler()),   # Data normalization
    ('pca', PCA(n_components=2))    # PCA
]

# Create the pipeline
pca_pipe = Pipeline(steps)

# Step 3
# Apply the pipeline over X_train
X_train_pca = pca_pipe.fit_transform(X_train)

# Apply the same pipeline over X_test
X_test_pca = pca_pipe.transform(X_test)

# Step 4
# Define the logistic regression classifier
logistic_clf = LogisticRegression()

# Create a pipeline with logistic regression
clf = Pipeline([
    ('logistic', logistic_clf)
])

# Now you can use clf for fitting and predicting

#Step 5
# Train clf with X_train_pca and y_train
clf.fit(X_train_pca, y_train)

# Step 6: Obtain predictions for X_test_pca
y_pred_pca = clf.predict(X_test_pca)

# Step 7: Create the decision boundary display using DecisionBoundaryDisplay()
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X_train_pca, response_method="predict",
    xlabel='X1', ylabel='X2',
    alpha=0.5
)

# Step 8: Display model
disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train)
plt.show()

# Calculate accuracy score of the model using only 2 components of PCA
accuracy = accuracy_score(y_test, y_pred_pca)
print('Accuracy: :', accuracy)

