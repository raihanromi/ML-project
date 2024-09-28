import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import joblib
from Preprocessing.data_preparation import X_train,X_test,y_train,y_test



print(f"X_train shape:{X_train.shape}")
print(f"y_train shape:{y_train.shape}")


model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64,activation='relu'),
    Dense(32,activation='relu'),
    Dense(4)
])


model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mae'])

model.fit(X_train,y_train,epochs=20,batch_size=10,validation_data=(X_test,y_test))

loss,mae = model.evaluate(X_test,y_test)

print(f" Test Loss : {loss} , TEST MAE: {mae}")

prediction= model.predict(X_test)


predicted_calories = prediction[0][0]
predicted_carbs = prediction[0][1]
predicted_protein = prediction[0][2]
predicted_fat = prediction[0][3]


print(f"Predicted Calories: {predicted_calories:.2f}g")
print(f"Predicted Carbs: {predicted_carbs:.2f}g")
print(f"Predicted Protein: {predicted_protein:.2f}g")
print(f"Predicted fat: {predicted_fat:.2f}g")

model.save('model/nutrition_model.h5')

print("Model training Compeleted")