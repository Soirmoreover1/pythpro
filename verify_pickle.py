import pickle

# Define the path to your pickle files
model_path = 'knn_model.pickle'
scaler_path = 'scaler.pickle'

# Function to load and verify pickle files
def verify_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Load and verify the KNN model
knn = verify_pickle(model_path)

# Load and verify the scaler
scaler = verify_pickle(scaler_path)