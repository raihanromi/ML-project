import numpy as np 
import joblib
import tensorflow as tf

model = tf.keras.models.load_model('server/model/nutrition_model.h5')

label_encoders = joblib.load('backend/server/model/label_encoders.pkl')
scaler  = joblib.load('server/model/scaler.pkl')

input_data={

            'age':30,
            'gender':'male',
            'activity_level':'moderate',
            'goal':'muscle gain',
            'weight':75,
            'height':180,
            'current_calories':2500,            
}

input_data['gender'] = label_encoders['gender'].transform([input_data['gender']])[0]
input_data['activity_level'] = label_encoders['activity_level'].transform([input_data['activity_level']])[0]
input_data['goal'] = label_encoders['goal'].transform([input_data['goal']])[0]

scaled_data = scaler.transform([[input_data['age'],input_data['weight'],input_data['height'],input_data['current_calories']]])

final_input = np.hstack((
    scaled_data, 
    np.array(input_data['gender']).reshape(1, 1), 
    np.array(input_data['activity_level']).reshape(1, 1), 
    np.array(input_data['goal']).reshape(1, 1)
)).reshape(1, -1)


print(final_input)

print('final input shape: {final_input.shape}')

try:
    prediction= model.predict(final_input)
    predicted_calories = prediction[0][0]
    predicted_carbs = prediction[0][1]
    predicted_protein = prediction[0][2]
    predicted_fat = prediction[0][3]


    print(f"Predicted Calories: {predicted_calories:.2f}g")
    print(f"Predicted Carbs: {predicted_carbs:.2f}g")
    print(f"Predicted Protein: {predicted_protein:.2f}g")
    print(f"Predicted fat: {predicted_fat:.2f}g")


except ValueError as e:
    print(f"Error during prediction:{e}")