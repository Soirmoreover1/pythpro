import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the dataset
df = pd.read_csv("Employee Data.csv")

# Encode the 'Gender' column
gender_encoded = pd.get_dummies(df['Gender'], drop_first=True)
df = pd.concat([df, gender_encoded], axis=1)

# Prepare the feature matrix and target vector
x = df[['Male', 'Satisfaction', 'Salary', 'Home-Office', 'Department']].to_numpy()
y = df['Attrition'].to_numpy()

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)

# Save the KNN model and scaler to pickle files
with open('knn_model.pickle', 'wb') as f:
    pickle.dump(knn, f)

with open('scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)

print("Pickle files created successfully.")