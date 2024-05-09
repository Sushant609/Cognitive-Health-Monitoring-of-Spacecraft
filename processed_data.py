import pandas as pd
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

# Load the dataset
data_set = pd.read_csv('flight_phase_test.csv')
data = data_set.copy()  

# Data pre-processing

# Apply predictive imputation
data = predictive_imputation(data)

# Relationship between independent variables(CORRELATION MATRIX)
data2 = data.dropna(axis=0)
correlation = data2.corr()
print(correlation)



