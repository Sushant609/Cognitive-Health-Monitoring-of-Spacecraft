import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
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

# Assuming all columns are relevant for conceptual anomaly detection
X = data[['TS', 'ALT', 'SPD', 'ROC']]

# Initialize the KMeans model
k = 4  # Number of clusters (you can adjust this based on your data)
model = KMeans(n_clusters=k, random_state=42)

# Fit the model
model.fit(X)

# Get cluster labels and cluster centers
cluster_labels = model.labels_
cluster_centers = model.cluster_centers_

# Calculate distances from each data point to its respective cluster center
distances = []
for i in range(len(X)):
    cluster_index = cluster_labels[i]
    center = cluster_centers[cluster_index]
    distance = np.linalg.norm(X.iloc[i] - center)
    distances.append(distance)

# Determine a threshold for anomaly detection
threshold = np.mean(distances) + 2 * np.std(distances)

# Identify conceptual anomalies based on the threshold
anomalies = data[distances > threshold]

# Visualize the data and anomalies
plt.figure(figsize=(10, 6))

# Scatter plot for ALT column
plt.subplot(2, 2, 1)
plt.scatter(data.index, data['ALT'], c=cluster_labels, cmap='viridis')
plt.scatter(anomalies.index, anomalies['ALT'], color='red', label='Contextual Anomalies')
plt.xlabel('Time')
plt.ylabel('Altitude')
plt.title('Altitude with Contextual Anomalies')
# Scatter plot for SPD column
plt.subplot(2, 2, 2)
plt.scatter(data.index, data['SPD'], c=cluster_labels, cmap='viridis')
plt.scatter(anomalies.index, anomalies['SPD'], color='red', label='Contextual Anomalies')
plt.xlabel('Time')
plt.ylabel('Speed')
plt.title('Speed with Contextual Anomalies')

# Scatter plot for ROC column
plt.subplot(2, 2, 3)
plt.scatter(data.index, data['ROC'], c=cluster_labels, cmap='viridis')
plt.scatter(anomalies.index, anomalies['ROC'], color='red', label='Contextual Anomalies')
plt.xlabel('Time')
plt.ylabel('Rate of Climb')
plt.title('Rate of Climb with Contextual Anomalies')

plt.tight_layout()
plt.legend()
plt.show()
