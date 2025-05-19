import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QFormLayout, 
    QPushButton, QMessageBox, QGroupBox, QSlider
)
from PyQt5.QtCore import Qt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost_model import XGBoostModel

class ObesityPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Load the best model (XGBoost) and encoders from main.py
        self.xgb_model = XGBoostModel(random_state=2021, use_gpu=False)
        self.load_model_and_encoders()
        
        # Define ranges and options
        self.gender_options = ["Female", "Male"]
        self.family_history_options = ["no", "yes"]
        self.favc_options = ["no", "yes"]
        self.caec_options = ["no", "Sometimes", "Frequently", "Always"]
        self.smoke_options = ["no", "yes"]
        self.scc_options = ["no", "yes"]
        self.mtrans_options = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
        self.calc_options = ["no", "Sometimes", "Frequently", "Always"]
        
        # Set up the UI
        self.setWindowTitle("Obesity Level Prediction")
        self.setMinimumSize(800, 600)
        
        # Create main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create form layout for inputs
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        
        # Create demographic group
        demographic_group = QGroupBox("Personal Information")
        demographic_layout = QFormLayout(demographic_group)
        
        # Add Gender
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(self.gender_options)
        demographic_layout.addRow("Gender:", self.gender_combo)
        
        # Add Age
        self.age_spin = QSpinBox()
        self.age_spin.setRange(10, 100)
        self.age_spin.setValue(30)
        demographic_layout.addRow("Age:", self.age_spin)
        
        # Add Height (m)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 2.5)
        self.height_spin.setValue(1.70)
        self.height_spin.setSingleStep(0.01)
        self.height_spin.setDecimals(2)
        demographic_layout.addRow("Height (m):", self.height_spin)
        
        # Add Weight (kg)
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(30.0, 250.0)
        self.weight_spin.setValue(70.0)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setDecimals(1)
        demographic_layout.addRow("Weight (kg):", self.weight_spin)
        
        # Add Family History
        self.family_history_combo = QComboBox()
        self.family_history_combo.addItems(self.family_history_options)
        demographic_layout.addRow("Family History with Overweight:", self.family_history_combo)
        
        # Create lifestyle group
        lifestyle_group = QGroupBox("Lifestyle Factors")
        lifestyle_layout = QFormLayout(lifestyle_group)
        
        # Add FAVC (frequent consumption of high caloric food)
        self.favc_combo = QComboBox()
        self.favc_combo.addItems(self.favc_options)
        lifestyle_layout.addRow("Frequent High Calorie Food:", self.favc_combo)
        
        # Add FCVC (frequency of vegetable consumption)
        self.fcvc_spin = QDoubleSpinBox()
        self.fcvc_spin.setRange(1.0, 3.0)
        self.fcvc_spin.setValue(2.0)
        self.fcvc_spin.setSingleStep(0.1)
        self.fcvc_spin.setDecimals(1)
        lifestyle_layout.addRow("Vegetable Consumption (1-3):", self.fcvc_spin)
        
        # Add NCP (number of main meals)
        self.ncp_spin = QDoubleSpinBox()
        self.ncp_spin.setRange(1.0, 4.0)
        self.ncp_spin.setValue(3.0)
        self.ncp_spin.setSingleStep(0.1)
        self.ncp_spin.setDecimals(1)
        lifestyle_layout.addRow("Number of Main Meals (1-4):", self.ncp_spin)
        
        # Add CAEC (consumption of food between meals)
        self.caec_combo = QComboBox()
        self.caec_combo.addItems(self.caec_options)
        lifestyle_layout.addRow("Food Between Meals:", self.caec_combo)
        
        # Add SMOKE
        self.smoke_combo = QComboBox()
        self.smoke_combo.addItems(self.smoke_options)
        lifestyle_layout.addRow("Do you smoke?", self.smoke_combo)
        
        # Add CH2O (water consumption)
        self.ch2o_spin = QDoubleSpinBox()
        self.ch2o_spin.setRange(1.0, 3.0)
        self.ch2o_spin.setValue(2.0)
        self.ch2o_spin.setSingleStep(0.1)
        self.ch2o_spin.setDecimals(1)
        lifestyle_layout.addRow("Water Consumption (1-3):", self.ch2o_spin)
        
        # Add SCC (calorie monitoring)
        self.scc_combo = QComboBox()
        self.scc_combo.addItems(self.scc_options)
        lifestyle_layout.addRow("Monitor Calories:", self.scc_combo)
        
        # Add FAF (physical activity frequency)
        self.faf_spin = QDoubleSpinBox()
        self.faf_spin.setRange(0.0, 3.0)
        self.faf_spin.setValue(1.0)
        self.faf_spin.setSingleStep(0.1)
        self.faf_spin.setDecimals(1)
        lifestyle_layout.addRow("Physical Activity (0-3):", self.faf_spin)
        
        # Add TUE (time using technology devices)
        self.tue_spin = QDoubleSpinBox()
        self.tue_spin.setRange(0.0, 2.0)
        self.tue_spin.setValue(1.0)
        self.tue_spin.setSingleStep(0.1)
        self.tue_spin.setDecimals(1)
        lifestyle_layout.addRow("Technology Use (0-2):", self.tue_spin)
        
        # Add CALC (alcohol consumption)
        self.calc_combo = QComboBox()
        self.calc_combo.addItems(self.calc_options)
        lifestyle_layout.addRow("Alcohol Consumption:", self.calc_combo)
        
        # Add MTRANS (transportation mode)
        self.mtrans_combo = QComboBox()
        self.mtrans_combo.addItems(self.mtrans_options)
        lifestyle_layout.addRow("Transportation Mode:", self.mtrans_combo)
        
        # Create prediction button
        self.predict_button = QPushButton("Predict Obesity Level")
        self.predict_button.clicked.connect(self.make_prediction)
        
        # Add groups to main layout
        main_layout.addWidget(demographic_group)
        main_layout.addWidget(lifestyle_group)
        main_layout.addWidget(self.predict_button)
        
        # Create result display
        self.result_label = QLabel("Enter your information and click Predict")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14pt; margin-top: 20px;")
        main_layout.addWidget(self.result_label)
        
        # Set the central widget
        self.setCentralWidget(central_widget)
    
    def load_model_and_encoders(self):
        """Load the trained model and encoders"""
        try:
            # Try to load saved model
            import pickle
            with open('xgb_model.pkl', 'rb') as f:
                self.xgb_model = pickle.load(f)
            with open('encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print("Loaded saved model and encoders")
        except:
            # If no saved model, train a new one on a smaller dataset
            print("No saved model found. Please train the model first.")
            QMessageBox.warning(
                self, 
                "Model Not Found", 
                "The prediction model was not found. Please run the main.py script first to train the model."
            )
            # We will use the model created in __init__ with default parameters
    
    def make_prediction(self):
        """Process input data and make a prediction"""
        try:
            # Get values from inputs
            gender = self.gender_combo.currentText()
            age = self.age_spin.value()
            height = self.height_spin.value()
            weight = self.weight_spin.value()
            family_history = self.family_history_combo.currentText()
            favc = self.favc_combo.currentText()
            fcvc = self.fcvc_spin.value()
            ncp = self.ncp_spin.value()
            caec = self.caec_combo.currentText()
            smoke = self.smoke_combo.currentText()
            ch2o = self.ch2o_spin.value()
            scc = self.scc_combo.currentText()
            faf = self.faf_spin.value()
            tue = self.tue_spin.value()
            calc = self.calc_combo.currentText()
            mtrans = self.mtrans_combo.currentText()
            
            # Calculate derived features
            bmi = weight / (height ** 2)
            hydration = ch2o / ncp
            calorie_burn_proxy = faf * weight
            
            # Create a dataframe with the input data
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Height': [height],
                'Weight': [weight], 
                'family_history_with_overweight': [family_history],
                'FAVC': [favc],
                'FCVC': [fcvc],
                'NCP': [ncp],
                'CAEC': [caec],
                'SMOKE': [smoke],
                'CH2O': [ch2o],
                'SCC': [scc],
                'FAF': [faf],
                'TUE': [tue],
                'CALC': [calc],
                'MTRANS': [mtrans],
                'BMI': [bmi],
                'Hydration': [hydration],
                'CalorieBurnProxy': [calorie_burn_proxy]
            })
            
            # Load the expected columns from the saved file if it exists
            try:
                with open('feature_columns.txt', 'r') as f:
                    expected_columns = f.read().splitlines()
                print(f"Model expects {len(expected_columns)} columns")
            except:
                # If no columns file, proceed with default approach
                expected_columns = None
                print("No feature columns file found, using default preprocessing")
            
            # Encode categorical features (using the same encoding as in main.py)
            # One-hot encode categorical features
            hot_encoders = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS']
            input_encoded = pd.get_dummies(input_data, columns=hot_encoders, drop_first=True)
            
            # Label encode other categorical features
            label_encoders = ['CAEC', 'CALC']
            for feature in label_encoders:
                if feature == 'CAEC':
                    mapping = {'no': 3, 'Sometimes': 2, 'Frequently': 1, 'Always': 0}
                elif feature == 'CALC':
                    mapping = {'no': 3, 'Sometimes': 2, 'Frequently': 1, 'Always': 0}
                
                input_encoded[feature] = input_data[feature].map(mapping)
            
            # Scale numerical features
            scale_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC', 'BMI', 'Hydration', 'CalorieBurnProxy']
            
            # If we have a saved scaler, use it
            if hasattr(self, 'scaler') and self.scaler is not None:
                for feature in scale_features:
                    if feature in input_encoded.columns:
                        # Apply the saved scaler's parameters
                        input_encoded[feature] = (input_encoded[feature] - self.scaler.mean_[scale_features.index(feature)]) / self.scaler.scale_[scale_features.index(feature)]
            else:
                # Simple standardization as fallback
                for feature in scale_features:
                    if feature in input_encoded.columns:
                        input_encoded[feature] = (input_encoded[feature] - 0) / 1
            
            # Ensure all required columns are present based on expected_columns
            if expected_columns:
                # Create a DataFrame with all expected columns, initialized to 0
                complete_input = pd.DataFrame(0, index=[0], columns=expected_columns)
                
                # Fill in the values we have
                for col in input_encoded.columns:
                    if col in expected_columns:
                        complete_input[col] = input_encoded[col].values
                
                # Use this as our input
                input_encoded = complete_input
                
                print(f"Prepared input with {len(input_encoded.columns)} columns")
            
            # Make prediction using the model
            if hasattr(self, 'xgb_model') and self.xgb_model is not None:
                prediction_value = self.xgb_model.predict(input_encoded)
                
                # Map prediction back to obesity level
                obesity_levels = [
                    "Insufficient Weight", 
                    "Normal Weight", 
                    "Obesity Type I", 
                    "Obesity Type II", 
                    "Obesity Type III", 
                    "Overweight Level I", 
                    "Overweight Level II"
                ]
                
                # Ensure prediction is an integer index
                if isinstance(prediction_value, (list, np.ndarray)):
                    pred_idx = int(prediction_value[0])
                else:
                    pred_idx = int(prediction_value)
                
                predicted_level = obesity_levels[pred_idx]
                
                # Update result label with color coding
                self.result_label.setText(f"Predicted Obesity Level: {predicted_level}")
                
                # Color code based on prediction
                if pred_idx == 0:  # Insufficient Weight
                    self.result_label.setStyleSheet("font-size: 14pt; margin-top: 20px; color: blue;")
                elif pred_idx == 1:  # Normal Weight
                    self.result_label.setStyleSheet("font-size: 14pt; margin-top: 20px; color: green;")
                elif pred_idx <= 4:  # Obesity
                    self.result_label.setStyleSheet("font-size: 14pt; margin-top: 20px; color: red;")
                else:  # Overweight
                    self.result_label.setStyleSheet("font-size: 14pt; margin-top: 20px; color: orange;")
                
                # Also show BMI value
                bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
                QMessageBox.information(
                    self,
                    "Prediction Results",
                    f"BMI: {bmi:.2f} ({bmi_category})\nPredicted Level: {predicted_level}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Model Not Available",
                    "The prediction model is not available. Please train the model first by running main.py."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ObesityPredictionApp()
    window.show()
    sys.exit(app.exec_()) 