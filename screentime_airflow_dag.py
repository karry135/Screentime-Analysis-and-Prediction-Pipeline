import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Load the Dataset

data = pd.read_csv('screentime_analysis.csv')

#Check for missing values and duplicates
print(data.isnull().sum())
print(data.duplicated().sum()) 

#Convert Date column to DateTime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month

#Encode the categorical 'App' column using one-hot encoding
data = pd.get_dummies(data, columns=['App'], drop_first=True)

scaler = MinMaxScaler()

#Scale numerical features using MinMaxScaler
data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

#Feature Engineering
data['Previous_Day_Usage'] = data['Usage (minutes)'].shift(1)
data['Notifications_x_TimesOpened'] = data['Notifications'] * data['Times Opened']

#Save the preprocessed data to a file
data.to_csv('preprocessed_screentime_analysis.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Step 1: Check for missing values in the feature matrix (X) and target variable (y)
print(X.isnull().sum())  # Features with missing values
print(y.isnull().sum())  # Target variable with missing values

# Step 2: Handle missing values
imputer = SimpleImputer(strategy='mean') 

# Apply imputer to fill missing values in X (features)
X_imputed = imputer.fit_transform(X)

# Step 3: Check again after imputation
print(pd.DataFrame(X_imputed).isnull().sum()) 

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)


# Step 5: Initialize and train the RandomForestRegressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
predictions = model.predict(X_test)

# Step 7: Evaluate the model using Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')


pip install apache-airflow

from sklearn.preprocessing import MinMaxScaler
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import logging  # For logging instead of print


# define the data preprocessing function
def preprocess_data():
    try:
        file_path = 'screentime_analysis.csv'
        data = pd.read_csv(file_path)

        # Data preprocessing steps
        data['Date'] = pd.to_datetime(data['Date'])
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['Month'] = data['Date'].dt.month

        data = data.drop(columns=['Date'])  # Drop the 'Date' column

        # One-hot encoding for 'App' column
        data = pd.get_dummies(data, columns=['App'], drop_first=True)

        # Scale the 'Notifications' and 'Times Opened' columns
        scaler = MinMaxScaler()
        data[['Notifications', 'Times Opened']] = scaler.fit_transform(data[['Notifications', 'Times Opened']])

        # Save the preprocessed data to a new CSV
        preprocessed_path = 'preprocessed_screentime_analysis.csv'
        data.to_csv(preprocessed_path, index=False)

        # Log success
        logging.info(f"Preprocessed data saved to {preprocessed_path}")
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")


# define the DAG
dag = DAG(
    dag_id='data_preprocessing',
    schedule_interval='@daily',  # Set the schedule to run daily
    start_date=datetime(2025, 1, 1),
    catchup=False,  # Don't backfill past dates
)
