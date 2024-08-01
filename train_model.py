import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('loan_data.csv')  # Make sure your CSV has relevant columns

# Feature selection
X = data[['credit_score', 'num_transactions', 'salary']]
y = data['loan_approved']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
