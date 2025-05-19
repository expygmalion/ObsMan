# Obesity Prediction Application

This application provides a user-friendly GUI for predicting obesity levels based on personal information and lifestyle factors. It uses machine learning models trained on health data to make accurate predictions.

## Features

- Predict obesity level based on personal information and lifestyle factors
- Calculate and display BMI 
- Interactive GUI for easy data input
- Trained using multiple ML models (XGBoost, SVM, Logistic Regression, KNN, Random Forest)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ObesityManagement
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the GUI Application

```bash
python run_app.py
```

This will launch the GUI application. If the model hasn't been trained yet, it will ask if you want to train it first.

### Option 2: Train the Model First

```bash
python save_model.py
```

This will train the XGBoost model and save it for future use.

### Option 3: Run the Main Analysis

```bash
python main.py
```

This will run the full data analysis and training of multiple models.

## Data Input Fields

- **Personal Information**: Gender, Age, Height, Weight, Family history of obesity
- **Lifestyle Factors**: 
  - Dietary habits (high-calorie food, vegetable consumption, meals per day)
  - Physical activity frequency
  - Water consumption
  - Technology use time
  - Transportation mode
  - Alcohol consumption
  - Smoking status

## Model Information

The application uses an XGBoost model for predictions, which was selected as the best performing model after comparison with SVM, Logistic Regression, KNN, and Random Forest. The model achieves approximately 99% accuracy on the test set.

## File Structure

- `main.py`: The main analysis script that processes data and trains models
- `obesity_prediction_gui.py`: The GUI application
- `run_app.py`: Launcher for the GUI application
- `save_model.py`: Script to train and save the model
- `xgboost_model.py`: Implementation of the XGBoost model
- Other model implementations: `svm_model.py`, `knn_model.py`, etc.

## Dependencies

- NumPy
- Pandas
- Scikit-learn
- XGBoost
- PyQt5
- Matplotlib
- Seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details. 