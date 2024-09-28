from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

model = None  


model_path = os.path.join(os.getcwd(), 'model/nutrition_model.h5')

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print('Model loaded successfully.')
    else:
        print(f"Model not found at: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Load label encoders and scaler
try:
    label_encoders = joblib.load('model/label_encoders.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    print(f"Error loading encoders or scaler: {e}")


def suggest_foods(calories, carbs, protein, fat):
    suggestions = []
    
    if calories < 1800:
        suggestions.append("Consider meals high in protein, like grilled chicken or fish")
    elif 1800 <= calories <= 2500:
        suggestions.append("Incorporate balanced meals with lean meats, whole grains, and vegetables")
    else:
        suggestions.append("Include calorie-dense foods like avocados, nuts, and seeds.")

    if carbs < 150:
        suggestions.append("Add complex carbohydrates like sweet potatoes, or oats")
    elif carbs > 200:
        suggestions.append("Reduce high-carb foods")

    if protein < 70:
        suggestions.append("Increase protein intake with options like eggs, tofu, or protein")
    elif protein > 120:
        suggestions.append("Maintain high-protein meals")

    if fat < 50:
        suggestions.append("Increase healthy fats with olive oil, nuts, and seeds")
    elif fat > 90:
        suggestions.append("Cut down on fat by reducing fried foods")

    return ' '.join(suggestions)


@app.route('/')
def hello_world():
    return 'Hello'


@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    data = request.get_json()

    user_info = {
        'age': data.get('age'),
        'weight': data.get('weight'),
        'height': data.get('height'),
        'gender': data.get('gender'),
        'goal': data.get('goal'),
        'activity_level': data.get('activity_level'),
        'current_calories': data.get('current_calories')
    }

    # Preprocess input data
    try:
        scaled_data = scaler.transform([[user_info['age'], user_info['weight'], user_info['height'], user_info['current_calories']]])

        gender_encoder = np.array([label_encoders['gender'].transform([user_info['gender']])[0]]).reshape(1, -1)
        activity_level_encoder = np.array([label_encoders['activity_level'].transform([user_info['activity_level']])[0]]).reshape(1, -1)
        goal_encoder = np.array([label_encoders['goal'].transform([user_info['goal']])[0]]).reshape(1, -1)

        final_input = np.hstack((scaled_data, gender_encoder, activity_level_encoder, goal_encoder))
    except Exception as e:
        return jsonify({'error': f"Error in preprocessing input: {e}"}), 500

    # Make prediction
    try:
        prediction = model.predict(final_input)

        predicted_calories = prediction[0][0]
        predicted_carbs = prediction[0][1]
        predicted_protein = prediction[0][2]
        predicted_fat = prediction[0][3]

        print(f"Calories: {predicted_calories}, Carbs: {predicted_carbs}, Protein: {predicted_protein}, Fat: {predicted_fat}")

        food_suggestion = suggest_foods(predicted_calories, predicted_carbs, predicted_protein, predicted_fat)

        return jsonify({
            'response': f"Suggested daily intake: {predicted_calories:.2f} calories, {predicted_carbs:.2f}g carbs, {predicted_protein:.2f}g protein, {predicted_fat:.2f}g fat",
            'food_suggestion': food_suggestion
        })
    except Exception as e:
        return jsonify({'error': f"Error in making prediction: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
