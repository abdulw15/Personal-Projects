# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet

# Section 2: Data Exploration
# Load the dataset into a Pandas dataframe.
data = pd.read_csv("Food Demand Forecasting Dataset.csv")

# Explore the dataset using summary statistics, visualizations, and other descriptive measures.
print(data.head())
print(data.describe())
print(data.info())

# Identify any patterns, trends, or anomalies in the data.
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
plt.title("Average Daily Demand by Month")
sns.lineplot(x="Month", y="Order_Demand", data=data.groupby("Month").mean().reset_index())
plt.show()

# Determine if any further data cleaning or preprocessing is required.
print(data.isnull().sum())

# Section 3: Data Preprocessing
# Clean and preprocess the data as needed.
data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
data["Day"] = data["Date"].dt.day
data = data.drop(["Date"], axis=1)

# Convert the data into a format suitable for modeling.
model_data = data.groupby(["Year", "Month"]).mean().reset_index()[["Year", "Month", "Order_Demand"]]
model_data.columns = ["year", "month", "order_demand"]
model_data["ds"] = pd.to_datetime(model_data[["year", "month"]].assign(day=1))
model_data = model_data.drop(["year", "month"], axis=1)

# Section 4: Modeling
# Create one or more time series models to forecast the demand for food delivery services.
# Split the dataset into training and testing sets.
train_size = int(len(model_data) * 0.8)
train_data = model_data[:train_size]
test_data = model_data[train_size:]

# Train the models on the training set and tune the hyperparameters.

## ARIMA Model
# Determine the optimal values of p, d, and q
p = range(0, 4)
d = range(0, 2)
q = range(0, 4)
pdq = [(x, y, z) for x in p for y in d for z in q]
aic = []
for param in pdq:
    try:
        model = ARIMA(train_data["order_demand"].to_numpy(), order=param)
        model_fit = model.fit()
        aic.append(model_fit.aic)
    except:
        continue
optimal = pdq[aic.index(min(aic))]

# Fit the ARIMA model on the training data
arima_model = ARIMA(train_data["order_demand"].to_numpy(), order=optimal)
arima_model_fit = arima_model.fit()

# Make predictions using the ARIMA model
arima_predictions = arima_model_fit.forecast(steps=len(test_data))

# Evaluate the performance of the ARIMA model
arima_mae = mean_absolute_error(test_data["order_demand"], arima_predictions)
arima_rmse = np.sqrt(mean_squared_error(test_data["order_demand"], arima_predictions))
print("ARIMA Model Evaluation:")
print("MAE: {:.2f}".format(arima_mae))
print("RMSE: {:.2f}".format(arima_rmse))

## Prophet Model
# Prepare the data in the required format for Prophet
prophet_data = model_data.rename(columns={"ds": "ds", "order_demand": "y"})

# Split the data into training and testing sets
prophet_train_data = prophet_data[:train_size]
prophet_test_data = prophet_data[train_size:]

# Fit the Prophet model on the training data
prophet_model = Prophet()
prophet_model.fit(prophet_train_data)

# Make predictions using the Prophet model
prophet_predictions = prophet_model.predict(prophet_test_data[["ds"]])
prophet_predictions = prophet_predictions["yhat"].to_numpy()

# Evaluate the performance of the Prophet model
prophet_mae = mean_absolute_error(prophet_test_data["y"].to_numpy(), prophet_predictions)
prophet_rmse = np.sqrt(mean_squared_error(prophet_test_data["y"].to_numpy(), prophet_predictions))
print("\nProphet Model Evaluation:")
print("MAE: {:.2f}".format(prophet_mae))
print("RMSE: {:.2f}".format(prophet_rmse))

# Select the best model based on the evaluation metrics
if arima_rmse < prophet_rmse:
    print("\nARIMA model is better than Prophet model for this dataset.")
else:
    print("\nProphet model is better than ARIMA model for this dataset.")

# Section 5: Visualization and Communication
# Create visualizations that communicate the findings and insights gained from the modeling

## Plot the actual and predicted values for ARIMA
arima_plot_data = pd.DataFrame({"actual": test_data["order_demand"], "predicted": arima_predictions}, index=test_data.index)
plt.figure(figsize=(10, 5))
plt.plot(train_data["order_demand"], label="Training data")
plt.plot(arima_plot_data["actual"], label="Actual values")
plt.plot(arima_plot_data["predicted"], label="Predicted values")
plt.title("ARIMA Model: Actual vs Predicted Order Demand")
plt.xlabel("Year")
plt.ylabel("Order Demand")
plt.legend()
plt.show()

## Plot the actual and predicted values for Prophet
prophet_plot_data = pd.DataFrame({"actual": prophet_test_data["y"].to_numpy(), "predicted": prophet_predictions}, index=prophet_test_data["ds"])
plt.figure(figsize=(10, 5))
plt.plot(prophet_train_data["ds"], prophet_train_data["y"], label="Training data")
plt.plot(prophet_plot_data["actual"], label="Actual values")
plt.plot(prophet_plot_data["predicted"], label="Predicted values")
plt.title("Prophet Model: Actual vs Predicted Order Demand")
plt.xlabel("Year")
plt.ylabel("Order Demand")
plt.legend()
plt.show()

## Plot the feature importance for the XGBoost model
xgb.plot_importance(xgb_model)
plt.title("Feature Importance for XGBoost Model")
plt.show()
