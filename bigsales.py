import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('data.csv')

# Print the dataset overview
print("Dataset Overview:")
print(data.head())

# Clean up column names (if needed)
data.columns = data.columns.str.strip()

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Optionally, drop or fill missing values here if any exist
# Example: data.dropna(inplace=True) or data.fillna(method='ffill', inplace=True)

# Define features and target
X = data[['feature1', 'feature2', 'feature3', 'feature4']]
y = data['target_column']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['feature1']
numerical_features = ['feature2', 'feature3', 'feature4']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Create a pipeline that first preprocesses the data and then fits the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Define hyperparameters for tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_features': ['sqrt', 'log2'],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nBest Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

# Get feature importances
regressor = best_model.named_steps['regressor']
importances = regressor.feature_importances_

# Get the names of the features from the OneHotEncoder
ohe_columns = best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)

# Combine feature names
feature_names = np.concatenate((numerical_features, ohe_columns))

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Save the model to a file
joblib.dump(best_model, 'random_forest_model.pkl')
