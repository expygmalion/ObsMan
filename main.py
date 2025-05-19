import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# Import our model classes
from xgboost_model import XGBoostModel
from svm_model import SVMModel
from logistic_regression_model import LogisticRegressionModel
from knn_model import KNNModel
from random_forest_model import RandomForestModel

# Load train and test datasets
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df.info()

test_df.isnull().sum()

train_df.isnull().sum()

# Convert target to numeric first
target_mapping = {cat: i for i, cat in enumerate(train_df['NObeyesdad'].unique())}
train_df['target_numeric'] = train_df['NObeyesdad'].map(target_mapping)

# Make sure test dataframe also has target_numeric for consistency
test_df['target_numeric'] = test_df['NObeyesdad'].map(target_mapping)

# Handle categorical features
data = train_df.copy()
for col in data.select_dtypes(include=['object']).columns:
    if col != 'NObeyesdad':
        # Create dummy variables and drop the original
        dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
        data = pd.concat([data, dummies], axis=1)
        data.drop(col, axis=1, inplace=True)

# Drop the original target column
if 'NObeyesdad' in data.columns:
    data.drop('NObeyesdad', axis=1, inplace=True)

# Calculate correlation with numeric target
correlations = data.corr()['target_numeric'].drop('target_numeric').sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(x=correlations.values[:15], y=correlations.index[:15])
plt.title('Top Features Correlated with Obesity')
plt.xlabel('Correlation')
plt.tight_layout()
plt.show()

# Simple FCVC plot for skewness check
plt.figure(figsize=(6, 4))
sns.histplot(train_df['FCVC'], kde=True)
plt.title('FCVC Distribution')
plt.show()

# Print skewness value
skew = train_df['FCVC'].skew()
print(f"FCVC Skewness: {skew}")

print("Sum of test duplicates", test_df.duplicated().sum())
print("Sum of train duplicates", train_df.duplicated().sum())

## Safely drop duplicates
train_df_duplicates_handled = train_df.drop_duplicates()
print(train_df_duplicates_handled.duplicated().sum())

# Fill FCVC with mode and drop CALC
mode_value = train_df_duplicates_handled['FCVC'].mode()[0]
train_df_duplicates_handled_filled_FCVC = train_df_duplicates_handled.fillna({'FCVC': mode_value})
print(train_df_duplicates_handled_filled_FCVC.isnull().sum())

calc_missing = train_df_duplicates_handled_filled_FCVC['CALC'].isnull().astype(int)

# Convert target to numeric
target_mapping = {cat: i for i, cat in enumerate(train_df_duplicates_handled_filled_FCVC['NObeyesdad'].unique())}
target_numeric = train_df_duplicates_handled_filled_FCVC['NObeyesdad'].map(target_mapping)

# Calculate correlation with target
target_corr = np.corrcoef(calc_missing, target_numeric)[0, 1]

# Print just the correlation number
print(f"{target_corr:.4f}")

train_df_duplicates_handled_filled_FCVC_dropped_CALC = train_df_duplicates_handled_filled_FCVC.dropna()
train_df_duplicates_handled_drop_missing = train_df_duplicates_handled_filled_FCVC_dropped_CALC
train_df_duplicates_handled_drop_missing.isnull().sum()

# Create a copy to avoid SettingWithCopyWarning
train_df_clean = train_df_duplicates_handled_drop_missing.copy()

# Add engineered features
train_df_clean['BMI'] = train_df_clean['Weight'] / (train_df_clean['Height'] ** 2)
train_df_clean['Hydration'] = train_df_clean['CH2O'] / train_df_clean['NCP']
train_df_clean['CalorieBurnProxy'] = train_df_clean['FAF'] * train_df_clean['Weight']
train_df_clean.head()

test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)
test_df['Hydration'] = test_df['CH2O'] / test_df['NCP']
test_df['CalorieBurnProxy'] = test_df['FAF'] * test_df['Weight']

features_numbers = ['Age','Height','Weight','BMI','Hydration','CalorieBurnProxy'] ## goes into standard scaling
features_nominal = ['Gender','family_history_with_overweight','FAVC','SMOKE','SCC','MTRANS']
feature_ordinal = ['FCVC','NCP','CH2O', 'CAEC', 'FAF','CALC', 'TUE', 'NObeyesdad' ]
label_encoders = ['CAEC', 'CALC', 'NObeyesdad']
hot_encoders = ['Gender','family_history_with_overweight','FAVC','SMOKE','SCC','MTRANS']
scale_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC','BMI','Hydration','CalorieBurnProxy' ]

# Handle outliers in numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(features_numbers):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=train_df_clean[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Cap outliers using the IQR method
for col in features_numbers:
    if pd.api.types.is_numeric_dtype(train_df_clean[col]):
        Q1 = train_df_clean[col].quantile(0.25)
        Q3 = train_df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Cap the outliers
        train_df_clean[col] = train_df_clean[col].clip(lower=lower_bound, upper=upper_bound)

# Create a grid of boxplots for each feature after capping outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(features_numbers):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=train_df_clean[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Class distribution plot
plt.figure(figsize=(10, 6))
sns.countplot(x='NObeyesdad', data=train_df_clean, 
              order=train_df_clean['NObeyesdad'].value_counts().index,
              palette='Set2')
plt.xticks(rotation=45)
plt.title('Obesity Levels Distribution')
plt.xlabel('Obesity Level')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric features
plt.figure(figsize=(12, 8))
corr = train_df_clean[features_numbers].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()

# Create a dictionary to store the label encoders
encoders = {}

# Iterate through each feature and encode it
for feature in label_encoders:
    # Create a new label encoder for each feature
    encoders[feature] = LabelEncoder()

    # Fit and transform the training data
    train_df_clean[feature] = encoders[feature].fit_transform(train_df_clean[feature])

    # Transform the test data using the same encoder
    test_df[feature] = encoders[feature].transform(test_df[feature])

    # Print the mapping for each feature
    print(f"\nMapping for {feature}:")
    for i, label in enumerate(encoders[feature].classes_):
        print(f"{label} → {i}")

x_train = train_df_clean.drop('NObeyesdad', axis=1)
y_train = train_df_clean['NObeyesdad']

x_test = test_df.drop('NObeyesdad', axis=1)
y_test = test_df['NObeyesdad']

print(x_train.head())
print(y_train.head())
print(x_test.head())
print(y_test.head())

# Drop target_numeric if present
if 'target_numeric' in x_train.columns:
    x_train = x_train.drop('target_numeric', axis=1)
if 'target_numeric' in x_test.columns:
    x_test = x_test.drop('target_numeric', axis=1)

# One-hot encode categorical features
x_train_encoded = pd.get_dummies(x_train, columns=hot_encoders, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=hot_encoders, drop_first=True)

print("Shape after one-hot encoding:")
print(f"x_train_encoded: {x_train_encoded.shape}")
print(f"x_test_encoded: {x_test_encoded.shape}")

# Standardize features for models that are sensitive to scale
scaler_std = StandardScaler()
x_train_encoded_std = x_train_encoded.copy()
x_test_encoded_std = x_test_encoded.copy()

# Fit on train, transform train and test
x_train_encoded_std[scale_features] = scaler_std.fit_transform(x_train_encoded[scale_features])
x_test_encoded_std[scale_features] = scaler_std.transform(x_test_encoded[scale_features])

# MinMax scaling for chi-square feature selection
scaler_minmax = MinMaxScaler()
x_train_encoded_minmax = x_train_encoded.copy()
x_test_encoded_minmax = x_test_encoded.copy()

x_train_encoded_minmax[scale_features] = scaler_minmax.fit_transform(x_train_encoded[scale_features])
x_test_encoded_minmax[scale_features] = scaler_minmax.transform(x_test_encoded[scale_features])

# Feature selection with Pearson correlation for SVM and KNN
x_train_encoded_std_for_selector = x_train_encoded_std.copy()
x_test_encoded_std_for_selector = x_test_encoded_std.copy()

selector_pearson = SelectKBest(score_func=f_classif, k=10)
x_train_pearson = selector_pearson.fit_transform(x_train_encoded_std_for_selector, y_train)
x_test_pearson = selector_pearson.transform(x_test_encoded_std_for_selector)

# Get the selected feature names
pearson_features = x_train_encoded_std_for_selector.columns[selector_pearson.get_support()]

# Convert to DataFrame
x_train_pearson = pd.DataFrame(x_train_pearson, columns=pearson_features)
x_test_pearson = pd.DataFrame(x_test_pearson, columns=pearson_features)

print("Top features selected by Pearson correlation:")
print(pearson_features)

# Feature selection with Chi-square for Logistic Regression
x_train_encoded_minmax_for_selector = x_train_encoded_minmax.copy()
x_test_encoded_minmax_for_selector = x_test_encoded_minmax.copy()

selector_chi2 = SelectKBest(score_func=chi2, k=10)
x_train_chi = selector_chi2.fit_transform(x_train_encoded_minmax_for_selector, y_train)
x_test_chi = selector_chi2.transform(x_test_encoded_minmax_for_selector)

# Get selected feature names
chi_features = x_train_encoded_minmax_for_selector.columns[selector_chi2.get_support()]

# Create DataFrames for selected features
x_train_chi = pd.DataFrame(x_train_chi, columns=chi_features)
x_test_chi = pd.DataFrame(x_test_chi, columns=chi_features)

print("Top features selected by Chi-square:")
print(chi_features)

# Reset indices to avoid mismatches
x_train_pearson.reset_index(drop=True, inplace=True)
x_test_pearson.reset_index(drop=True, inplace=True)
x_train_chi.reset_index(drop=True, inplace=True)
x_test_chi.reset_index(drop=True, inplace=True)

# Check for high correlation in Pearson selected features
plt.figure(figsize=(12,10))
mask = np.triu(np.ones_like(x_train_pearson[pearson_features].corr(), dtype=bool))
sns.heatmap(x_train_pearson[pearson_features].corr(),
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1, vmax=1,
            mask=mask,
            square=True)
plt.title("Pearson-Selected Feature Correlation (Upper Triangle)", pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Remove highly correlated features from Pearson selection
if 'Weight' in x_train_pearson.columns:
    x_train_pearson.drop('Weight', axis=1, inplace=True)
    x_test_pearson.drop('Weight', axis=1, inplace=True)
    pearson_features = pearson_features.drop('Weight')

print("Pearson features after removing high correlation:")
print(pearson_features)

# Check for high correlation in Chi-square selected features
plt.figure(figsize=(12,10))
mask = np.triu(np.ones_like(x_train_chi[chi_features].corr(), dtype=bool))
sns.heatmap(x_train_chi[chi_features].corr(),
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-1, vmax=1,
            mask=mask,
            square=True)
plt.title("Chi-Square Selected Feature Correlation (Upper Triangle)", pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Remove highly correlated features from Chi-square selection
if 'Weight' in x_train_chi.columns:
    x_train_chi.drop('Weight', axis=1, inplace=True)
    x_test_chi.drop('Weight', axis=1, inplace=True)
    chi_features = chi_features.drop('Weight')

print("Chi-square features after removing high correlation:")
print(chi_features)

# Check distance distribution for KNN
from scipy.spatial.distance import pdist, squareform
dists = pdist(x_train_pearson.values, metric='euclidean')
plt.hist(dists, bins=50)
plt.title("Pairwise Euclidean Distances")
plt.show()

# Check class balance
plt.figure(figsize=(8, 6))
y_train.value_counts(normalize=True).plot(kind='bar')
plt.title("Class Distribution")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()

# Check scaling metrics
print("Standard scaling stats:")
print(x_train_encoded_std[scale_features].describe().T[['mean', 'std']])

# Check skewness
print("Skewness after scaling:")
print(x_train_encoded_std[scale_features].skew())

# Model training and evaluation
print("\n========== Training and Evaluating Models ==========\n")

# XGBoost Model with GPU acceleration
print("\n--- XGBoost Model (CPU) ---\n")
xgb_model = XGBoostModel(random_state=2021, use_gpu=False)  # Disable GPU since CuPy is not installed
xgb_model.train(x_train_encoded, y_train)
xgb_results = xgb_model.evaluate(x_train_encoded, y_train, x_test_encoded, y_test)

print(f"XGBoost Train Accuracy: {xgb_results['train_accuracy']*100:.2f}%")
print(f"XGBoost Test Accuracy: {xgb_results['test_accuracy']*100:.2f}%")
print(f"XGBoost Train Precision: {xgb_results['train_precision']*100:.2f}%")
print(f"XGBoost Test Precision: {xgb_results['test_precision']*100:.2f}%")
print(f"XGBoost Train Recall: {xgb_results['train_recall']*100:.2f}%")
print(f"XGBoost Test Recall: {xgb_results['test_recall']*100:.2f}%")
print(f"XGBoost Train F1: {xgb_results['train_f1']*100:.2f}%")
print(f"XGBoost Test F1: {xgb_results['test_f1']*100:.2f}%")
print(f"XGBoost Training Time: {xgb_results['training_time']:.2f} seconds")
print(f"XGBoost Prediction Time: {xgb_results['prediction_time']:.2f} seconds")
print("XGBoost Confusion Matrix:")
print(xgb_results['test_confusion_matrix'])

# SVM Model
print("\n--- SVM Model ---\n")
svm_model = SVMModel()
svm_model.train(x_train_pearson, y_train)
svm_results = svm_model.evaluate(x_train_pearson, y_train, x_test_pearson, y_test)

print(f"SVM Train Accuracy: {svm_results['train_accuracy']*100:.2f}%")
print(f"SVM Test Accuracy: {svm_results['test_accuracy']*100:.2f}%")
print(f"SVM Train Precision: {svm_results['train_precision']*100:.2f}%")
print(f"SVM Test Precision: {svm_results['test_precision']*100:.2f}%")
print(f"SVM Train Recall: {svm_results['train_recall']*100:.2f}%")
print(f"SVM Test Recall: {svm_results['test_recall']*100:.2f}%")
print(f"SVM Train F1: {svm_results['train_f1']*100:.2f}%")
print(f"SVM Test F1: {svm_results['test_f1']*100:.2f}%")
print("SVM Confusion Matrix:")
print(svm_results['test_confusion_matrix'])

# Logistic Regression Model
print("\n--- Logistic Regression Model ---\n")
lr_model = LogisticRegressionModel()
lr_model.train(x_train_chi, y_train)
lr_results = lr_model.evaluate(x_train_chi, y_train, x_test_chi, y_test)

print(f"LR Train Accuracy: {lr_results['train_accuracy']*100:.2f}%")
print(f"LR Test Accuracy: {lr_results['test_accuracy']*100:.2f}%")
print(f"LR Train Precision: {lr_results['train_precision']*100:.2f}%")
print(f"LR Test Precision: {lr_results['test_precision']*100:.2f}%")
print(f"LR Train Recall: {lr_results['train_recall']*100:.2f}%")
print(f"LR Test Recall: {lr_results['test_recall']*100:.2f}%")
print(f"LR Train F1: {lr_results['train_f1']*100:.2f}%")
print(f"LR Test F1: {lr_results['test_f1']*100:.2f}%")
print("LR Confusion Matrix:")
print(lr_results['test_confusion_matrix'])

# KNN Model
print("\n--- KNN Model ---\n")
knn_model = KNNModel()
knn_model.train(x_train_pearson, y_train)
knn_results = knn_model.evaluate(x_train_pearson, y_train, x_test_pearson, y_test)

print(f"KNN Train Accuracy: {knn_results['train_accuracy']*100:.2f}%")
print(f"KNN Test Accuracy: {knn_results['test_accuracy']*100:.2f}%")
print(f"KNN Train Precision: {knn_results['train_precision']*100:.2f}%")
print(f"KNN Test Precision: {knn_results['test_precision']*100:.2f}%")
print(f"KNN Train Recall: {knn_results['train_recall']*100:.2f}%")
print(f"KNN Test Recall: {knn_results['test_recall']*100:.2f}%")
print(f"KNN Train F1: {knn_results['train_f1']*100:.2f}%")
print(f"KNN Test F1: {knn_results['test_f1']*100:.2f}%")
print("KNN Confusion Matrix:")
print(knn_results['test_confusion_matrix'])

# Random Forest Model
print("\n--- Random Forest Model ---\n")
rf_model = RandomForestModel()
rf_model.train(x_train_encoded, y_train)
rf_results = rf_model.evaluate(x_train_encoded, y_train, x_test_encoded, y_test)

print(f"RF Train Accuracy: {rf_results['train_accuracy']*100:.2f}%")
print(f"RF Test Accuracy: {rf_results['test_accuracy']*100:.2f}%")
print(f"RF Train Precision: {rf_results['train_precision']*100:.2f}%")
print(f"RF Test Precision: {rf_results['test_precision']*100:.2f}%")
print(f"RF Train Recall: {rf_results['train_recall']*100:.2f}%")
print(f"RF Test Recall: {rf_results['test_recall']*100:.2f}%")
print(f"RF Train F1: {rf_results['train_f1']*100:.2f}%")
print(f"RF Test F1: {rf_results['test_f1']*100:.2f}%")
print("RF Confusion Matrix:")
print(rf_results['test_confusion_matrix'])

# Compare models
print("\n========== Model Comparison ==========\n")
model_names = ['XGBoost', 'SVM', 'Logistic Regression', 'KNN', 'Random Forest']
test_accuracies = [
    xgb_results['test_accuracy'],
    svm_results['test_accuracy'],
    lr_results['test_accuracy'],
    knn_results['test_accuracy'],
    rf_results['test_accuracy']
]

# Plot model comparison
plt.figure(figsize=(12, 6))
colors = ['blue', 'green', 'red', 'purple', 'orange']
bars = plt.bar(model_names, [acc * 100 for acc in test_accuracies], color=colors)

# Add the accuracy values on top of the bars
for bar, accuracy in zip(bars, test_accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f'{accuracy * 100:.2f}%',
        ha='center',
        fontweight='bold'
    )

plt.title('Model Comparison - Test Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nBest performing model based on test accuracy:")
best_model_index = test_accuracies.index(max(test_accuracies))
print(f"  {model_names[best_model_index]} - {test_accuracies[best_model_index]*100:.2f}%") 