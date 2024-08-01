from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    credit_score = float(request.form['credit_score'])
    salary = float(request.form['salary'])
    
    # Dummy value for num_transactions
    num_transactions = 50  # Adjust or obtain this value as needed

    features = np.array([[credit_score, num_transactions, salary]])
    prediction = model.predict(features)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'

    return render_template('index.html', prediction_text=f'Loan Status: {result}')

if __name__ == '__main__':
    app.run(debug=True)
