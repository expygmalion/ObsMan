import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost_model import XGBoostModel

# This script should be run after main.py to save the trained model

def save_trained_model():
    print("Loading and processing data...")
    
    # Load train and test datasets
    train_df = pd.read_csv('train_dataset.csv')
    test_df = pd.read_csv('test_dataset.csv')
    
    # Convert target to numeric
    target_mapping = {cat: i for i, cat in enumerate(train_df['NObeyesdad'].unique())}
    train_df['target_numeric'] = train_df['NObeyesdad'].map(target_mapping)
    test_df['target_numeric'] = test_df['NObeyesdad'].map(target_mapping)
    
    # Add engineered features
    train_df['BMI'] = train_df['Weight'] / (train_df['Height'] ** 2)
    train_df['Hydration'] = train_df['CH2O'] / train_df['NCP']
    train_df['CalorieBurnProxy'] = train_df['FAF'] * train_df['Weight']
    
    test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)
    test_df['Hydration'] = test_df['CH2O'] / test_df['NCP']
    test_df['CalorieBurnProxy'] = test_df['FAF'] * test_df['Weight']
    
    # Handle missing values
    train_df = train_df.dropna()
    
    # Create a dictionary to store encoders
    encoders = {}
    
    # Label encode categorical features
    label_encoders = ['CAEC', 'CALC', 'NObeyesdad']
    for feature in label_encoders:
        encoders[feature] = LabelEncoder()
        train_df[feature] = encoders[feature].fit_transform(train_df[feature])
        if feature in test_df.columns:
            test_df[feature] = encoders[feature].transform(test_df[feature])
    
    x_train = train_df.drop(['NObeyesdad', 'target_numeric'], axis=1, errors='ignore')
    y_train = train_df['NObeyesdad']
    
    # One-hot encode categorical features
    hot_encoders = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']
    x_train_encoded = pd.get_dummies(x_train, columns=hot_encoders, drop_first=True)
    
    # Features to scale
    scale_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC', 'BMI', 'Hydration', 'CalorieBurnProxy']
    
    # Standardize numerical features
    scaler = StandardScaler()
    x_train_encoded[scale_features] = scaler.fit_transform(x_train_encoded[scale_features])
    
    print("Training XGBoost model...")
    # Train a XGBoost model (simplified version)
    xgb_model = XGBoostModel(random_state=2021, use_gpu=False)
    xgb_model.train(x_train_encoded, y_train)
    
    print("Saving model and preprocessing components...")
    # Save model, encoders, and scaler
    with open('xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save column names for reference
    with open('feature_columns.txt', 'w') as f:
        f.write('\n'.join(x_train_encoded.columns))
    
    print("Model and preprocessing components saved successfully.")
    print("Now you can run the obesity_prediction_gui.py application.")

if __name__ == "__main__":
    save_trained_model() 