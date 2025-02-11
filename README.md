Screentime Analysis and Prediction Pipeline
This project implements an end-to-end pipeline for processing and predicting screentime data. It covers data preprocessing, feature engineering, model training, and automation using Apache Airflow.

Key Steps:
Data Preprocessing:

Loads screentime data from a CSV file.
Cleans the data by handling missing values and duplicates.
Converts the Date column to datetime, extracting DayOfWeek and Month.
One-hot encodes the App column and normalizes Notifications and Times Opened.
Creates additional features such as Previous_Day_Usage and Notifications_x_TimesOpened.
Model Training:

Defines features (X) and target variable (y).
Handles missing values with a SimpleImputer and splits the data into training and testing sets.
Trains a RandomForestRegressor to predict screentime usage (Usage (minutes)).
Evaluates the model using Mean Absolute Error (MAE).
Automation with Apache Airflow:

Integrates Apache Airflow to schedule and automate the data preprocessing pipeline.
Runs daily to ensure up-to-date processing and saves cleaned data.
