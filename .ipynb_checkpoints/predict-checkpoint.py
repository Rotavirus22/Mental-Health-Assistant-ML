from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    # Load the model using joblib
   model = pickle.load(open('StackingClassifierManual.pkl','rb')) # Adjust the path if necessary
except Exception as e:
    print(f"Error loading model: {e}")
    
# Mapping user responses to numerical values
response_mapping = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

# Define the questions to serve to users
def generate_questions():
    return [
        {
            "id": "Anxiety_Level",
            "question": "How frequently do you feel anxious, nervous, or on edge?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "id": "Depression_Symptoms",
            "question": "How often do you feel down, depressed, or hopeless?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "id": "Stress_Frequency",
            "question": "How often do you feel overwhelmed or unable to cope with your daily tasks?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "id": "Sleep_Issues",
            "question": "How often do you experience trouble sleeping due to stress or anxiety?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        },
        {
            "id": "Fatigue_Level",
            "question": "How frequently do you feel tired or have low energy?",
            "options": ["Never", "Rarely", "Sometimes", "Often", "Always"]
        }
    ]

@app.route('/questions', methods=['GET'])
def get_questions():
    """Return a list of mental health questions."""
    return jsonify(generate_questions())

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on user responses."""
    data = request.get_json()

    # Validate input data
    if not data or 'responses' not in data:
        return jsonify({'error': 'Invalid input: No responses provided'}), 400

    user_responses = data['responses']

    # Convert responses to numerical values
    try:
        input_features = [response_mapping[response] for response in user_responses]
    except KeyError as e:
        return jsonify({'error': f'Invalid response: {str(e)}'}), 400

    input_data = np.array(input_features).reshape(1, -1)

    # Make a prediction using the loaded model
    try:
        prediction = model.predict(input_data)
    except Exception as e:
        return jsonify({'error': f"Model prediction error: {str(e)}"}), 500

    # Interpret the prediction result
    condition = "Mental Health Condition Present" if prediction[0] == 1 else "No Mental Health Condition"
    return jsonify({'prediction': condition})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
