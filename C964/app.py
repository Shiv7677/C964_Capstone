from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from scipy.stats import binom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the models
model1 = joblib.load("neutral_nonneutral_classification_model.pkl")
model2 = joblib.load("driver_prediction_model.pkl")

# Load the dataset (if needed for analysis)
data = pd.read_csv(r"C:\C964\DNA_mutation_dataset\combined_dataset_with_recalculated_q_values.csv")


def calculate_b_score(n, N, p):
    """
    Calculate B_Score using the binomial cumulative distribution.
    Parameters:
    - n: Observed mutations.
    - N: Total samples (count).
    - p: Mutability (probability of mutation for the context).

    Returns:
    - B_Score
    """
    try:
        b_score = 1 - binom.cdf(n, N, p)  # Complement of CDF to calculate P(X > n)
        return b_score
    except Exception as e:
        raise ValueError(f"Error in calculating B_Score: {str(e)}")


def predict_pipeline(gene, mutation, mutability, count, b_score=None, threshold1=0.53, threshold2=0.54):
    """
    Predict whether a mutation is:
    - Neutral
    - Non-Neutral + Cancer Driver
    - Non-Neutral + Potential Driver
    - Non-Neutral + Passenger
    """
    try:
        # Step 1: Prepare input for Model 1
        input_model1 = pd.DataFrame({
            'gene': [gene],
            'mutation': [mutation],
            'mutability': [mutability],
            'count': [count]
        })
        input_model1 = pd.get_dummies(input_model1).reindex(columns=model1.feature_names_in_, fill_value=0)

        # Step 2: Predict Neutral/Non-Neutral with Model 1
        is_non_neutral = model1.predict(input_model1)[0]

        if not is_non_neutral:  # If neutral, return result
            return {"Result": "Neutral"}

        # Step 3: Calculate or use provided B_Score
        if b_score is None:
            b_score = calculate_b_score(count, count, mutability)

        # Step 4: Classify Driver Status using Threshold Logic
        if b_score < threshold1:
            driver_status = "Cancer Driver"
        elif threshold1 <= b_score <= threshold2:
            driver_status = "Potential Driver"
        else:
            driver_status = "Passenger"

        # Step 5: Prepare input for Model 2
        input_model2 = pd.DataFrame({
            'mutability': [mutability],
            'count': [count],
            'B_Score': [b_score]
        })

        # Step 6: Predict Driver/Non-Driver with Model 2
        is_driver = model2.predict(input_model2)[0]
        probability = model2.predict_proba(input_model2)[0][1]  # Driver probability

        # Ensure consistency: Passenger cannot be driver
        if driver_status == "Passenger":
            is_driver = 0
            probability = 0.0

        # Step 7: Return Final Result
        return {
            "Result": "Non-Neutral",
            "Driver Status": driver_status,
            "Driver Prediction": "Driver" if is_driver else "Non-Driver",
            "Driver Probability": probability
        }

    except Exception as e:
        return {"error": f"Error in prediction pipeline: {str(e)}"}


@app.route('/')
def home():
    """Render the main page."""
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Parse user input
        gene = request.form['gene']
        mutation = request.form['mutation']
        mutability = float(request.form['mutability'])
        count = int(request.form['count'])
        b_score = float(request.form['b_score']) if 'b_score' in request.form and request.form['b_score'] else None

        # Run the prediction pipeline
        result = predict_pipeline(gene, mutation, mutability, count, b_score)

        # Render the template with the result
        return render_template("index.html", result=result)

    except Exception as e:
        # Handle errors and render the error message
        error_message = {"error": str(e)}
        return render_template("index.html", result=error_message)


@app.route('/feature_importance')
def feature_importance():
    """Render the feature importance visualization."""
    return '''
    <div class="text-center">
        <h4>Feature Importance for Driver Prediction</h4>
        <img src="static/images/feature_importance.png" class="img-fluid mt-3" alt="Feature Importance">
    </div>
    '''


@app.route('/confusion_matrix_static')
def confusion_matrix_static():
    """Render the confusion matrix visualization."""
    return '''
    <div class="text-center">
        <h4>Confusion Matrix for Neutral/Non-Neutral Classification</h4>
        <img src="static/images/confusion_matrix.png" class="img-fluid mt-3" alt="Confusion Matrix">
    </div>
    '''


@app.route('/correlation_matrix')
def correlation_matrix():
    """Render the correlation matrix visualization."""
    return '''
    <div class="text-center">
        <h4>Correlation Matrix for All Numeric Features for Both Models</h4>
        <img src="static/images/correlation_matrix.png" class="img-fluid mt-3" alt="Correlation Matrix">
    </div>
    '''


if __name__ == "__main__":
    app.run(debug=True)
