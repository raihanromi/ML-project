import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def load_or_create_dataset(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f'Loading dataset from {filepath}')
        return pd.read_csv(filepath)
    else:
        print('Using mock dataset as no filepath was provided or found')
        data = {
            'age':np.random.randint(18,65,100),
            'gender':np.random.choice(['male','female'],100),
            'activity_level':np.random.choice(['low','moderate','high'],100),
            'goal':np.random.choice(['weight loss','muscle gain','maintance'],100),
            'weight':np.random.randint(50,120,100),
            'height':np.random.randint(150,200,100),
            'current_calories':np.random.randint(1200,3500,100),
            
            'suggested_calories': np.random.randint(1200, 3500, 100),
            'suggested_carbs':np.random.uniform(100,400,100),
            'suggested_protein':np.random.uniform(50,150,100),
            'suggested_fat':np.random.uniform(40,100,100)

        }

        return pd.DataFrame(data)
    

df = load_or_create_dataset()
df = df.dropna()

lable_encoders ={}
for column in [ 'gender','activity_level','goal']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    lable_encoders[column]=le

scaler = StandardScaler()

df[['age','weight','height','current_calories']] = scaler.fit_transform(df[['age','weight','height','current_calories']])

x = df[['age','gender', 'activity_level','goal','weight','height','current_calories']]
y = df[['suggested_calories', 'suggested_carbs', 'suggested_protein', 'suggested_fat']]


X_train,X_test,y_train,y_test  = train_test_split(x,y,test_size=0.2,random_state=42)

os.makedirs('model',exist_ok=True)

joblib.dump(lable_encoders,'model/label_encoders.pkl')
joblib.dump(scaler,'model/scaler.pkl')


X_train.to_csv('model/X_train.csv',index=False)
X_test.to_csv('model/X_test.csv',index=False)
y_train.to_csv('model/y_train.csv',index=False)
y_test.to_csv('model/y_test.csv',index=False)

print("Data  preparation completed and preprocessing objects saved")

