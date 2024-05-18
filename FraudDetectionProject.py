import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd

# Load the CSV file into a Pandas DataFrame
unscaled_df = pd.read_csv("creditcard.csv")

unscaled_df.shape
unscaled_df.info()
unscaled_df.head(5)


# Exploratory Data Analysis
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
correlation_matrix = unscaled_df.corr()

# Create a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",annot_kws={"size": 7})
plt.title('Correlation Matrix Heatmap')
plt.show()

normal_df = unscaled_df[unscaled_df['Class']==0]
fraud_df = unscaled_df[unscaled_df['Class']==1]

# plot of high value transactions
bins = np.linspace(200, 2500, 100)
plt.figure(figsize=(14, 8))
plt.hist(normal_df.Amount, bins, alpha=1, density=True, label='Normal')
plt.hist(fraud_df.Amount, bins, alpha=0.6, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions \$50
          +)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)");
plt.show()

# Observation: There are very few fraud cases among a lot of regular transactions, the data looks more scattered and unpredictable. Especially in the cases where there are very few frauds, it's tough to tell them apart just by looking at how much money was involved in each transaction.

bins = np.linspace(0, 48, 48) #48 hours
plt.figure(figsize=(14, 8))
plt.hist((normal_df.Time/(60*60)), bins, alpha=1, density=True, label='Normal')
plt.hist((fraud_df.Time/(60*60)), bins, alpha=0.6, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Percentage of transactions by hour")
plt.xlabel("Transaction time as measured from first transaction in the dataset (hours)")
plt.ylabel("Percentage of transactions (%)");
plt.show()

Observation: Think of hour "zero" as when the very first transaction occurred, which might not necessarily mean midnight. Looking at the pattern of regular transactions, there's a noticeable drop around hours 1 to 8 and again around hours 24 to 32. This suggests that these times might be nighttime in this dataset. If this holds true, it seems like fraud happens more often during these nighttime hours. It's not straightforward to build a clear-cut classifier based solely on time differences between regular and fraudulent transactions.

# Feature Engineering
from sklearn.preprocessing import MinMaxScaler

# Initialize the StandardScaler
scaler = MinMaxScaler()

# Apply min-max scaling to the entire DataFrame
scaled_values = scaler.fit_transform(unscaled_df)
df = pd.DataFrame(scaled_values, columns=unscaled_df.columns)

# Display the first 5 rows of the scaled DataFrame
print(df.head())

Transformations to perform: drop 'Time' columns as it was starting from t=0 to t=ti which does not have any effect on the classification.

# Drop unnecessary columns
df.drop(['Time'],axis=1,inplace=True)
df.head(5)

# Separate features and target variable
X = df.drop('Class', axis=1)  # Adjust 'target_column_name' to your target column
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Train a classifier (Example: RandomForestClassifier)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the classifier
y_pred_rf = clf.predict(X_test)
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Calculate false positive rate (fpr) and true positive rate (tpr) for ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_rf)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Create a confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

Accuracy: 0.9995201479348805
Precision: 0.8321678321678322
Recall: 0.875
F1-score: 0.8530465949820789

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression classifier
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_resampled, y_train_resampled)

# Evaluation Metrics Logistic Regression
# Evaluate the classifier
y_pred_lr = log_reg.predict(X_test)
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate false positive rate (fpr) and true positive rate (tpr) for ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_lr)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Create a confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

Accuracy: 0.975469026134382
Precision: 0.05615942028985507
Recall: 0.9117647058823529
F1-score: 0.10580204778156996

# Artificial Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# Define your ANN architecture
model = Sequential([
    Dense(64, input_shape=(X_train_resampled.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the resampled data
model.fit(X_train_resampled, y_train_resampled, epochs=10, batch_size=32, validation_split=0.1)

# Predict probabilities on the test set
y_pred_prob = model.predict(X_test)

# Convert probabilities to classes using a threshold (e.g., 0.5)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy:.4f}")

# Evaluation Metrics Artificil Neural Network
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

Precision: 0.0908
Recall: 0.9191
F1-score: 0.1652

# Tuning the Hyperparameters- Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2']  # Penalty norm
}

# Create the Logistic Regression model
lr = LogisticRegression(random_state=42, max_iter=1000)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(lr, param_grid, cv=3, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the Logistic Regression model with the best hyperparameters on the full training set
best_lr = LogisticRegression(random_state=42, max_iter=1000, **best_params)
best_lr.fit(X_train_resampled, y_train_resampled)

# Predict on the test set using the best model
y_pred_lr = best_lr.predict(X_test)

# Calculate evaluation metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
precision_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

# Print the evaluation metrics
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"Precision: {precision_lr:.4f}")
print(f"Recall: {recall_lr:.4f}")
print(f"F1-score: {f1_lr:.4f}")

# Best Hyperparameters: {'C': 100, 'penalty': 'l2'}
Accuracy: 0.9730
Precision: 0.0521
Recall: 0.9265
F1-score: 0.0986

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve-Adjusted')
plt.legend()
plt.show()

# Create a confusion matrix heatmap
conf_matrix = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap -Adjusted')
plt.show()

import matplotlib.pyplot as plt

# ROC AUC values
models = ['Logistic Regression', 'Random Forest', 'ANN', 'Adjusted Logistic Regression']
auc_values = [0.94, 0.94, 0.95, 0.94]

# Create bar chart
plt.figure(figsize=(8, 6))
plt.bar(models, auc_values, color=['blue', 'green', 'orange', 'red'])
plt.xlabel('Models')
plt.ylabel('ROC AUC')
plt.title('ROC AUC Values for Different Models')
plt.ylim(0.9, 1.0)  # Set y-axis limit for better visualization
plt.show()

import matplotlib.pyplot as plt

# Model names
models = ['Random Forest', 'Logistic Regression', 'ANN', 'Adjusted Logistic Regression']

# Metrics values for each model
accuracy = [0.99952, 0.975469, 0.9852, 0.9730]
precision = [0.832168, 0.056159, 0.0908, 0.521]
recall = [0.875, 0.911765, 0.9191, 0.9265]
f1_score = [0.853047, 0.105802, 0.1652, 0.0986]

# Plotting lines for each metric
plt.figure(figsize=(10, 6))

plt.plot(models, accuracy, marker='o', label='Accuracy')
plt.plot(models, precision, marker='o', label='Precision')
plt.plot(models, recall, marker='o', label='Recall')
plt.plot(models, f1_score, marker='o', label='F1-score')

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Metrics Value')
plt.title('Performance Metrics Comparison for Different Models')
plt.legend()
plt.grid(True)

# Show the line chart
plt.show()

