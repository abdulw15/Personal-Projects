# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Reviews.csv')

# Drop irrelevant columns and duplicate rows
data = data[['ProductId', 'UserId', 'Score', 'Summary', 'Text']]
data = data.drop_duplicates(subset=['UserId', 'ProductId', 'Text'])

# Convert the text data into numerical data
vectorizer = CountVectorizer()
text_data = vectorizer.fit_transform(data['Text'])

# Scale the Score column
scaler = StandardScaler()
data['Score'] = scaler.fit_transform(data[['Score']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_data, data['Score'], test_size=0.2, random_state=42)

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Create a list of models to evaluate
models = [
    LinearRegression(),
    RandomForestRegressor(),
    SVR()
]

# Evaluate the models using cross-validation
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{type(model).__name__}: {mean_squared_error(y_test, y_pred)}")

# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for the Random Forest Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
rf_model = RandomForestRegressor()
cv_model = GridSearchCV(rf_model, param_grid, scoring='neg_mean_squared_error')
cv_model.fit(X_train, y_train)
print(cv_model.best_params_)

# Import necessary libraries
from sklearn.metrics import r2_score

# Evaluate the performance of the final model on the test set
best_model = cv_model.best_estimator_
y_pred = best_model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred)}")
