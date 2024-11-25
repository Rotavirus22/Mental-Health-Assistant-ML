from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('manual_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Mapping user responses to numerical values
response_mapping = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5
}

# Gender and Age Mapping
gender_mapping = {
    "Male": 0,
    "Female": 1,
    "Other": 2
}

age_mapping = {
    "<18": 0,
    "18-30": 1,
    "31-50": 2,
    "50+": 3
}

# Yes/No Mapping
yes_no_mapping = {
    "Yes": 1,
    "No": 0
}

# Define the 8 questions to serve to users
def generate_questions():
    return [
        {
            "id": "Age",
            "question": "What is your age?",
            "options": ["<18", "18-30", "31-50", "50+"]
        },
        {
            "id": "Gender",
            "question": "What is your gender?",
            "options": ["Male", "Female", "Other"]
        },
        {
            "id": "family_history",
            "question": "Do you have a family history of mental health issues?",
            "options": ["Yes", "No"]
        },
        {
            "id": "benefits",
            "question": "Do you have access to mental health benefits at work?",
            "options": ["Yes", "No"]
        },
        {
            "id": "care_options",
            "question": "Do you have access to mental health care options?",
            "options": ["Yes", "No"]
        },
        {
            "id": "anonymity",
            "question": "Do you feel your mental health is protected in terms of privacy?",
            "options": ["Yes", "No"]
        },
        {
            "id": "leave",
            "question": "Does your employer offer leave for mental health issues?",
            "options": ["Yes", "No"]
        },
        {
            "id": "work_interfere",
            "question": "Does your work interfere with your mental health?",
            "options": ["Yes", "No"]
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
        # Apply gender and age mapping
        user_responses[1] = gender_mapping[user_responses[1]]  # Gender mapping
        user_responses[0] = age_mapping[user_responses[0]]  # Age mapping

        # Apply Yes/No mapping to the appropriate responses
        user_responses[2] = yes_no_mapping[user_responses[2]]  # Family history
        user_responses[3] = yes_no_mapping[user_responses[3]]  # Benefits
        user_responses[4] = yes_no_mapping[user_responses[4]]  # Care options
        user_responses[5] = yes_no_mapping[user_responses[5]]  # Anonymity
        user_responses[6] = yes_no_mapping[user_responses[6]]  # Leave
        user_responses[7] = yes_no_mapping[user_responses[7]]  # Work interfere

        # Convert other responses (e.g., "Never"/"Always") to numerical values using response_mapping
        input_features = [response_mapping.get(response, response) for response in user_responses]
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
