from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the saved KNN model and scaler
with open('knn_model.pickle', 'rb') as f:
    knn = pickle.load(f)

with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return "Welcome to the Employee Attrition Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read new data from the request
        new_data = request.get_json()
        new_df = pd.DataFrame(new_data)

        # Encode the 'Gender' column
        gender_encoded_new = pd.get_dummies(new_df['Gender'], drop_first=True)

        # Concatenate the encoded column to the new data
        df_new = pd.concat([new_df, gender_encoded_new], axis=1)

        # Prepare the feature matrix
        x_new = df_new[['Male', 'Satisfaction', 'Salary', 'Home-Office', 'Department']].to_numpy()

        # Scale the features using the loaded scaler
        x_new_scaled = scaler.transform(x_new)

        # Predict attrition for the new data
        y_new_pred = knn.predict(x_new_scaled)

        # Add the prediction results to the dataframe
        df_new['will_Attrition'] = y_new_pred

        # Convert the dataframe to JSON
        result = df_new.to_json(orient="records")
        return result

    except Exception as e:
        return str(e)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        # Get the data from the request
        row_values = request.get_json()
        
        # Convert to 2D NumPy array
        x_new_single = np.array(row_values).reshape(1, -1)

        # Scale the new row using the loaded scaler
        x_new_single_scaled = scaler.transform(x_new_single)

        # Predict attrition for the single new row
        y_new_single_pred = knn.predict(x_new_single_scaled)
        prediction = str(y_new_single_pred[0])
        return jsonify({"Attrition": prediction})

    except Exception as e:
        return str(e)

if __name__ == 'main':
    app.run(debug=True)