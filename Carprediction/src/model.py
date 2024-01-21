# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model(input_path, output_path):
    # Load processed data
    data = pd.read_csv('cardekho_data.csv')

    # Assuming 'features' and 'target' columns in your dataset
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model (replace with your actual model)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy}')

    # Save the trained model
    joblib.dump(model, output_path)

# Example usage:
# train_model('../data/processed_car_data.csv', '../models/car_price_model.pkl')
