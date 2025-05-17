# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

# Load train and test datasets
train_df = pd.read_csv('/home/expygmalion/Projects/PythonProjects/ObestiyManagement/train_dataset.csv')
test_df = pd.read_csv('/home/expygmalion/Projects/PythonProjects/ObestiyManagement/test_dataset.csv')

# %%
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_df

# %%
train_df.info()

# %%
test_df.isnull().sum()


# %%
train_df.isnull().sum()

# %%
print(test_df.duplicated().sum())

# %%
print(train_df.duplicated().sum())

# %%
train_df_duplicates_handled = train_df.drop_duplicates()

# %%
print(train_df_duplicates_handled.duplicated().sum())

# %%
train_df_duplicates_handled_drop_missing = train_df_duplicates_handled.dropna()

# %%
train_df_duplicates_handled_drop_missing.isnull().sum()

# %% [markdown]
# train_df_duplicates_handled_drop_missing

# %%
train_df_duplicates_handled_drop_missing['BMI'] = train_df_duplicates_handled_drop_missing['Weight'] / (train_df_duplicates_handled_drop_missing['Height'] ** 2)
train_df_duplicates_handled_drop_missing['Hydration'] = train_df_duplicates_handled_drop_missing['CH2O'] / train_df_duplicates_handled_drop_missing['NCP']
train_df_duplicates_handled_drop_missing['CalorieBurnProxy'] = train_df_duplicates_handled_drop_missing['FAF'] * train_df_duplicates_handled_drop_missing['Weight']
train_df_duplicates_handled_drop_missing.head()


# %% [markdown]
# ### Engineered features

# %%
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)
test_df['Hydration'] = test_df['CH2O'] / test_df['NCP']
test_df['CalorieBurnProxy'] = test_df['FAF'] * test_df['Weight']



# %%


# %% [markdown]
# loop in dtype.integer

# %%
features_numbers = ['Age','Height','Weight','BMI','Hydration','CalorieBurnProxy'] ## goes into standard scaling
features_nominal = ['Gender','family_history_with_overweight','FAVC','SMOKE','SCC','MTRANS'] ## goes into one hot encoding
feature_ordinal = ['FCVC','NCP','CH2O', 'CAEC', 'FAF','CALC', 'TUE', 'NObeyesdad' ] ## goes into label encoding
label_encoders = ['CAEC', 'CALC', 'NObeyesdad']
hot_encoders = ['Gender','family_history_with_overweight','FAVC','SMOKE','SCC','MTRANS'] 
scale_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC','BMI','Hydration','CalorieBurnProxy' ]



# %%
import pandas as pd

# %% [markdown]
# Before Dropping Outliers:

# %%
plt.figure(figsize=(15, 10))
for i, col in enumerate(features_numbers):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=train_df_duplicates_handled_drop_missing[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# %% [markdown]
# After Dropping Outliers:
# 

# %%
for col in features_numbers:
        if pd.api.types.is_numeric_dtype(train_df_duplicates_handled_drop_missing[col]):
            Q1 = train_df_duplicates_handled_drop_missing[col].quantile(0.25)
            Q3 = train_df_duplicates_handled_drop_missing[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap the outliers
            train_df_duplicates_handled_drop_missing[col] = train_df_duplicates_handled_drop_missing[col].clip(lower=lower_bound, upper=upper_bound)
            train_df=train_df_duplicates_handled_drop_missing[col]


# %%


# Create a grid of boxplots for each feature
plt.figure(figsize=(15, 10))
for i, col in enumerate(features_numbers):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=train_df_duplicates_handled_drop_missing[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Encoding:

# %%
from sklearn.preprocessing import LabelEncoder

# List of features to encode
label_encoders = ['CAEC', 'CALC', 'NObeyesdad']

# Create a dictionary to store the label encoders
encoders = {}

# Iterate through each feature and encode it
for feature in label_encoders:
    # Create a new label encoder for each feature
    encoders[feature] = LabelEncoder()
    
    # Fit and transform the training data
    train_df_duplicates_handled_drop_missing[feature] = encoders[feature].fit_transform(train_df_duplicates_handled_drop_missing[feature])
    
    # Transform the test data using the same encoder
    test_df[feature] = encoders[feature].transform(test_df[feature])
    
    # Print the mapping for each feature
    print(f"\nMapping for {feature}:")
    for i, label in enumerate(encoders[feature].classes_):
        print(f"{label} → {i}")

        train_df_duplicates_handled_drop_missing


# %%
x_train  = train_df_duplicates_handled_drop_missing.drop('NObeyesdad', axis=1)
y_train = train_df_duplicates_handled_drop_missing['NObeyesdad']

x_test = test_df.drop('NObeyesdad', axis=1)
y_test = test_df['NObeyesdad']

print(x_train.head())
print(y_train.head())
print(x_test.head())
print(y_test.head())



# %% [markdown]
# ## Hot Encoding for the rest of the features

# %%
x_train_encoded = pd.get_dummies(x_train, columns=hot_encoders, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=hot_encoders, drop_first=True)

# %%
x_train_encoded.head()

# %%
x_test_encoded.head()

# %%
y_test.head()

# %%
y_train.head()

# %% [markdown]
# ### StandardScaler – zero mean, unit variance

# %%
from sklearn.preprocessing import StandardScaler

scaler_std = StandardScaler()
x_train_encoded_std = x_train_encoded.copy()
x_test_encoded_std = x_test_encoded.copy()

# Fit on train, transform train
x_train_encoded_std[scale_features] = scaler_std.fit_transform(x_train_encoded[scale_features])
# Transform test
x_test_encoded_std[scale_features] = scaler_std.transform(x_test_encoded[scale_features])


x_train_encoded_std.head()

# %% [markdown]
# ### MinMaxScaler – scale to range [0, 1]

# %%
from sklearn.preprocessing import MinMaxScaler

scaler_minmax = MinMaxScaler()
x_train_encoded_minmax = x_train_encoded.copy()
x_test_encoded_minmax = x_test_encoded.copy()

x_train_encoded_minmax[scale_features] = scaler_minmax.fit_transform(x_train_encoded[scale_features])
x_test_encoded_minmax[scale_features] = scaler_minmax.transform(x_test_encoded[scale_features])

x_train_encoded_minmax.head()


# %%
x_train_encoded

# %% [markdown]
# ## Feature selection

# %% [markdown]
# ### 1. Pearson for KNN, Linear, SVM

# %%
from sklearn.feature_selection import SelectKBest, f_classif

# Use f_classif which relies on correlation and works for classification problems
selector_pearson = SelectKBest(score_func=f_classif, k=10)


x_train_pearson = selector_pearson.fit_transform(x_train_encoded_std, y_train)
x_test_pearson = selector_pearson.transform(x_test_encoded_std)

# Get the selected feature names
pearson_features = x_train_encoded_std.columns[selector_pearson.get_support()]

# Convert to DataFrame
x_train_pearson = pd.DataFrame(x_train_pearson, columns=pearson_features)
x_test_pearson = pd.DataFrame(x_test_pearson, columns=pearson_features)


x_test_pearson


# %%
pearson_features

# %% [markdown]
# ### 2. Chi-squared for Logistic Regression

# %%

from sklearn.feature_selection import SelectKBest, chi2


selector = SelectKBest(score_func=chi2, k=10)
x_train_chi = selector.fit_transform(x_train_encoded_minmax, y_train)
x_test_chi = selector.transform(x_test_encoded_minmax)

# Get selected feature names
chi_features = x_train_encoded_minmax.columns[selector.get_support()]

# Create DataFrames for selected f
# x_train_chieatures
x_train_chi = pd.DataFrame(x_train_chi, columns=chi_features)
x_test_chi = pd.DataFrame(x_test_chi, columns=chi_features)

x_train_chi

# %%
chi_features

# %%
#  index after DataFrame transformations
# To avoid index mismatches later in training/plotting:
x_train_pearson.reset_index(drop=True, inplace=True)
x_test_pearson.reset_index(drop=True, inplace=True)
x_train_chi.reset_index(drop=True, inplace=True)
x_test_chi.reset_index(drop=True, inplace=True)


# %% [markdown]
# # Setting the features ready for modeling:
# 

# %% [markdown]
#  We aim to validate the data's readiness for training by imposing checks and verifying some metrics relevant to each of the four select models we are to build (SVM, KNN, Linear, Logistic).
# 

# %% [markdown]
# ## Feature Correlation check

# %%
pearson_features

# %%
import numpy as np 
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

# %% [markdown]
# We must drop the highly correlated features first:

# %%
x_train_pearson.drop('Weight', axis=1, inplace=True)
x_test_pearson.drop('Weight', axis=1, inplace=True)
pearson_features = pearson_features.drop('Weight')


pearson_features

# %%
chi_features

# %%
import numpy as np 
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

# %% [markdown]
# Observation: we must drop the highly correlated features.

# %%
x_train_chi.drop('Weight', axis=1, inplace=True)
x_test_chi.drop('Weight', axis=1, inplace=True)
chi_features= chi_features.drop('Weight')


pearson_features

# %% [markdown]
# ## Distance Check for KNN 

# %%
from scipy.spatial.distance import pdist, squareform
dists = pdist(x_train_pearson.values, metric='euclidean')
plt.hist(dists, bins=50)
plt.title("Pairwise Euclidean Distances")
plt.show()


# %% [markdown]
# Observation: Bell shaped, no narrow peak.

# %% [markdown]
# ## Class Balance Check 
# We check for imbalanced classes that would potentially bias model's prediction.

# %%
y_train.value_counts(normalize=True).plot(kind='bar')

# %% [markdown]
#  Observation: All classes have < 10–15% class weights.

# %% [markdown]
# ## Scaling check

# %% [markdown]
# SVM and KNN are sensitive to distance metrics. Scaling ensures no feature dominates.

# %%
x_train_encoded_std[scale_features].describe().T[['mean', 'std']]

# %% [markdown]
# Observation: All means close to 0, all stds are ~1.

# %% [markdown]
#  ## Skewness Check 
# 
# %% [markdown]
# For Linear/Logistics: 
# Extreme skewness can degrade model performance. 

# %%
x_train_encoded_std[scale_features].skew()


# %%

# %% [markdown]
# ## Training Linear Regression Model

# %%
# Start model training section

# %% [markdown]
# # Model Training and Evaluation
# This section trains multiple models, optimizes their hyperparameters, and evaluates their performance.

# %% [markdown]
# ## 1. KNN Model

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Define hyperparameters for KNN
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize GridSearchCV
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
knn_grid.fit(x_train_pearson, y_train)

# Print best hyperparameters
print("Best KNN Hyperparameters:")
for param, value in knn_grid.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on test set
y_pred_knn = knn_grid.predict(x_test_pearson)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"\nKNN Test Accuracy: {knn_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 2. SVM Model

# %%
from sklearn.svm import SVC

# Define hyperparameters for SVM
svm_params = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear']
}

# Initialize GridSearchCV
svm_grid = GridSearchCV(
    SVC(probability=True),
    svm_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
svm_grid.fit(x_train_pearson, y_train)

# Print best hyperparameters
print("Best SVM Hyperparameters:")
for param, value in svm_grid.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on test set
y_pred_svm = svm_grid.predict(x_test_pearson)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"\nSVM Test Accuracy: {svm_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 3. Logistic Regression Model

# %%
from sklearn.linear_model import LogisticRegression

# Define hyperparameters for Logistic Regression
logreg_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2', None],
    'solver': ['lbfgs', 'newton-cg', 'sag']
}

# Initialize GridSearchCV
logreg_grid = GridSearchCV(
    LogisticRegression(max_iter=1000, multi_class='multinomial'),
    logreg_params,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
logreg_grid.fit(x_train_chi, y_train)

# Print best hyperparameters
print("Best Logistic Regression Hyperparameters:")
for param, value in logreg_grid.best_params_.items():
    print(f"  {param}: {value}")

# Evaluate on test set
y_pred_logreg = logreg_grid.predict(x_test_chi)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"\nLogistic Regression Test Accuracy: {logreg_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_logreg))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## 4. Linear Regression Model

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize and train Linear Regression
lr = LinearRegression()
lr.fit(x_train_pearson, y_train)

# Make predictions
y_pred_lr = lr.predict(x_test_pearson)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_lr)

# Print hyperparameters (Linear Regression doesn't have tunable hyperparameters)
print("Linear Regression Parameters:")
print("  fit_intercept: True")
print("  normalize: False")

# Print evaluation metrics
print("\nLinear Regression Metrics:")
print(f"  Mean Squared Error: {mse:.4f}")
print(f"  Root Mean Squared Error: {rmse:.4f}")
print(f"  R-squared Score: {r2:.4f}")

# Convert regression predictions to classes for classification metrics
y_pred_lr_classes = np.round(y_pred_lr).astype(int)
# Ensure predictions are within valid range
y_pred_lr_classes = np.clip(y_pred_lr_classes, 0, max(y_test))

# Calculate classification metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr_classes)
print(f"\nLinear Regression (Rounded) Classification Accuracy: {lr_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm_lr = confusion_matrix(y_test, y_pred_lr_classes)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Linear Regression (Rounded) Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## Model Comparison

# %%
# Compare accuracies of all models
model_accuracies = {
    'KNN': knn_accuracy,
    'SVM': svm_accuracy,
    'Logistic Regression': logreg_accuracy,
    'Linear Regression': lr_accuracy
}

# Plot comparison
plt.figure(figsize=(12, 6))
bars = plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

# Add accuracy values on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{list(model_accuracies.values())[i]:.4f}',
             ha='center', va='bottom')

plt.show()

# Print best model
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"Best performing model: {best_model} with accuracy {model_accuracies[best_model]:.4f}")

# %% [markdown]
# ## Best Model Hyperparameters Summary

# %%
print("Best Hyperparameters for Each Model:")
print("\nKNN:")
for param, value in knn_grid.best_params_.items():
    print(f"  {param}: {value}")

print("\nSVM:")
for param, value in svm_grid.best_params_.items():
    print(f"  {param}: {value}")

print("\nLogistic Regression:")
for param, value in logreg_grid.best_params_.items():
    print(f"  {param}: {value}")

print("\nLinear Regression:")
print("  No tunable hyperparameters")