import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function for treating missing and null values (PREDICTIVE IMPUTATION)
def predictive_imputation(data):

    columns_with_null = data.columns[data.isnull().any()]

    # Iterate over columns with missing values
    for col in columns_with_null:
        # Split data into features and target variable
        X = data.dropna().drop(columns_with_null, axis=1)  # Drop rows with null values and columns with missing values
        y = data.dropna()[col]  # Drop rows with null values and keep only the column with missing values
        
        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the linear regression model
        model = LinearRegression()

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Predict missing values in the test set
        predicted_values = model.predict(X_test)

        # Fill null values in the original dataset with predicted values
        data.loc[data[col].isnull(), col] = predicted_values

    return data

# Load the flight dataset
data_set = pd.read_csv('flight_phase_test.csv')
data = data_set.copy()  

# Data pre-processing

# Apply predictive imputation
data = predictive_imputation(data)

# Relationship between independent variables(CORRELATION MATRIX)
data2 = data.dropna(axis=0)
correlation = data2.corr()
print(correlation)


# Assuming all columns are relevant for anomaly detection
X = data[['TS', 'ALT', 'SPD', 'ROC']]

# Initialize the Isolation Forest model
model = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset

# Fit the model
model.fit(X)

# Predict outliers
outliers = model.predict(X)

# Add a column to indicate whether each data point is an outlier
data['is_outlier'] = outliers

# Visualize the data and anomalies
plt.figure(figsize=(10, 6))

# Scatter plot for ALT column
plt.subplot(2, 2, 1)
plt.scatter(data.index, data['ALT'], c=data['is_outlier'], cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Altitude')
plt.title('Altitude with Anomalies Detected')

# Scatter plot for SPD column
plt.subplot(2, 2, 2)
plt.scatter(data.index, data['SPD'], c=data['is_outlier'], cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Speed with Anomalies Detected')

# Scatter plot for ROC column
plt.subplot(2, 2, 3)
plt.scatter(data.index, data['ROC'], c=data['is_outlier'], cmap='viridis')
plt.xlabel('Time')
plt.ylabel('Rate of Climb')
plt.title('Rate of Climb with Anomalies Detected')

plt.tight_layout()
plt.show()











class RegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(3, 1)  # Output size is set to 1 for regression

    def forward(self, x):
        return self.linear(x)


# Initialize the regression model
regression_model = RegressionModel()

# Set optimizer and loss function
optimizer = torch.optim.SGD(regression_model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train the regression model
def train_regression(model, data, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

train_regression(regression_model, input_data, optimizer, criterion)

# Train the classification model
def train_classification(model, data, target, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

train_classification(classification_model, input_data, target, optimizer, criterion)


