from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

app = Flask(__name__)

# Load the pre-trained model
loaded_model = joblib.load("trained_model.pkl")

# Load the original training data for debugging
training_data = pd.read_csv("cardekho_data.csv")  # Replace with your actual training data file
X_train = training_data.drop(['Selling_Price'], axis=1)
X_train = pd.get_dummies(X_train, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

# Get the column names after one-hot encoding
input_data_columns = X_train.columns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract features from the form
        features = request.form.to_dict()
        print(features)
        features.pop('submit', None)  # Remove the submit button
        input_data = pd.DataFrame([features])

        # Ensure one-hot encoding consistency
        input_data = pd.get_dummies(input_data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)

        # Reindex the input data to have consistent column names
        input_data = input_data.reindex(columns=input_data_columns, fill_value=0)

        # Convert input_data to a NumPy array before making predictions
        prediction = loaded_model.predict(input_data.values)[0]
        print(prediction)

        return render_template('index.html', prediction_text=f'Predicted Selling Price: {prediction:.2f} Lakh INR')

if __name__ == '__main__':
    app.run(debug=True)
