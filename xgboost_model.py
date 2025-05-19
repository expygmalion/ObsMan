from xgboost import XGBClassifier, DMatrix, train
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import numpy as np
import pandas as pd
# Conditional import for cupy to make it optional

class XGBoostModel:
    def __init__(self, random_state=2021, use_gpu=True):
        self.random_state = random_state
        self.use_gpu = use_gpu
        # Try to import cupy if GPU is enabled
        if self.use_gpu:
            try:
                import cupy
                self.has_cupy = True
            except ImportError:
                print("Warning: CuPy module not found. Disabling GPU acceleration.")
                self.use_gpu = False
                self.has_cupy = False
        else:
            self.has_cupy = False
        
        self.model = None
        self.training_time = 0
        self.best_params = None
        self.feature_names = None
        
    def train(self, x_train, y_train):
        """Train XGBoost model with RandomizedSearchCV for hyperparameter tuning
        
        Args:
            x_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        # Save feature names if DataFrame
        if isinstance(x_train, pd.DataFrame):
            self.feature_names = x_train.columns.tolist()
            x_train_values = x_train.values
        else:
            x_train_values = x_train
            
        # Convert target to numpy array if needed
        if isinstance(y_train, pd.Series):
            y_train_values = y_train.values
        else:
            y_train_values = y_train
            
        # Base XGBoost parameters with hist method (updated for 2.0+)
        xgb_params = {
            'random_state': self.random_state,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',  # Use hist for both CPU and GPU
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train_values))
        }
        
        # Configure GPU acceleration
        if self.use_gpu:
            try:
                # Set device parameter (new recommended approach)
                xgb_params.update({
                    'device': 'cuda',  # Use CUDA instead of gpu_hist
                })
                print("XGBoost: Using GPU acceleration with CUDA")
            except Exception as e:
                print(f"Warning: Error configuring GPU for XGBoost: {e}")
                print("XGBoost: Falling back to CPU")
                self.use_gpu = False
                xgb_params.update({
                    'device': 'cpu',
                })
        else:
            # Explicitly set CPU device
            xgb_params.update({
                'device': 'cpu',
            })
        
        # Hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
        
        best_score = 0
        best_params = None
        best_model = None
        
        # Start training time measurement
        start_time = time.time()
        
        if self.use_gpu:
            try:
                # Prepare DMatrix for training
                dtrain = DMatrix(data=x_train_values, label=y_train_values)
                
                # Perform manual parameter tuning with cross-validation
                n_iter = 10  # Number of parameter combinations to try
                np.random.seed(self.random_state)
                
                for i in range(n_iter):
                    # Sample parameters
                    current_params = {
                        'max_depth': np.random.choice(param_grid['max_depth']),
                        'learning_rate': np.random.choice(param_grid['learning_rate']),
                        'n_estimators': np.random.choice(param_grid['n_estimators']),
                        'subsample': np.random.choice(param_grid['subsample']),
                        'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
                        'min_child_weight': np.random.choice(param_grid['min_child_weight'])
                    }
                    
                    # Update base parameters
                    params = {**xgb_params, **current_params}
                    
                    # Train with cross-validation
                    cv_results = train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=current_params['n_estimators'],
                        nfold=5,
                        early_stopping_rounds=10,
                        seed=self.random_state,
                        verbose_eval=False
                    )
                    
                    # Get best score
                    best_cv_score = cv_results['test-mlogloss-mean'].min()
                    
                    # Update best parameters if better
                    if best_params is None or -best_cv_score > best_score:  # Negative because we want to maximize
                        best_score = -best_cv_score
                        best_params = current_params
                
                # Train final model with best parameters
                final_params = {**xgb_params, **best_params}
                self.model = train(params=final_params, dtrain=dtrain, num_boost_round=best_params['n_estimators'])
                self.best_params = best_params
                
            except Exception as e:
                print(f"Error during GPU training: {e}")
                print("Falling back to standard RandomizedSearchCV...")
                self.use_gpu = False
                # Fall back to standard training method
                base_clf = XGBClassifier(**xgb_params)
                self.model = RandomizedSearchCV(
                    estimator=base_clf,
                    param_distributions=param_grid,
                    n_iter=10,
                    cv=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                self.model.fit(x_train_values, y_train_values)
                self.best_params = self.model.best_params_
        else:
            # Standard CPU training with RandomizedSearchCV
            base_clf = XGBClassifier(**xgb_params)
            self.model = RandomizedSearchCV(
                estimator=base_clf,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(x_train_values, y_train_values)
            self.best_params = self.model.best_params_
            
        # Calculate training time
        self.training_time = time.time() - start_time
        
        print(f"XGBoost training completed in {self.training_time:.2f} seconds")
        print(f"Best parameters: {self.best_params}")
        
        return self.model
    
    def evaluate(self, x_train, y_train, x_test, y_test):
        """Evaluate the model on training and test data"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Prepare data
        if isinstance(x_train, pd.DataFrame):
            x_train_values = x_train.values
        else:
            x_train_values = x_train
            
        if isinstance(x_test, pd.DataFrame):
            x_test_values = x_test.values
        else:
            x_test_values = x_test
            
        if isinstance(y_train, pd.Series):
            y_train_values = y_train.values
        else:
            y_train_values = y_train
            
        if isinstance(y_test, pd.Series):
            y_test_values = y_test.values
        else:
            y_test_values = y_test
        
        # Measure prediction time
        start_time = time.time()
        
        # Make predictions
        if self.use_gpu and not hasattr(self.model, 'predict'):
            # Using native XGBoost model
            dtrain = DMatrix(data=x_train_values, label=y_train_values, feature_names=self.feature_names)
            dtest = DMatrix(data=x_test_values, label=y_test_values, feature_names=self.feature_names)
            
            train_preds = self.model.predict(dtrain)
            test_preds = self.model.predict(dtest)
            
            # Convert predictions to integers if needed
            train_preds = train_preds.astype(int)
            test_preds = test_preds.astype(int)
        else:
            # Using scikit-learn API
            train_preds = self.model.predict(x_train_values)
            test_preds = self.model.predict(x_test_values)
        
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        results = {}
        
        # Accuracy
        results['train_accuracy'] = accuracy_score(y_train_values, train_preds)
        results['test_accuracy'] = accuracy_score(y_test_values, test_preds)
        
        # Precision
        results['train_precision'] = precision_score(y_train_values, train_preds, average='weighted')
        results['test_precision'] = precision_score(y_test_values, test_preds, average='weighted')
        
        # Recall
        results['train_recall'] = recall_score(y_train_values, train_preds, average='weighted')
        results['test_recall'] = recall_score(y_test_values, test_preds, average='weighted')
        
        # F1 Score
        results['train_f1'] = f1_score(y_train_values, train_preds, average='weighted')
        results['test_f1'] = f1_score(y_test_values, test_preds, average='weighted')
        
        # Confusion Matrix
        results['test_confusion_matrix'] = confusion_matrix(y_test_values, test_preds)
        
        # Timing information
        results['training_time'] = self.training_time
        results['prediction_time'] = prediction_time
        
        print(f"Prediction completed in {prediction_time:.2f} seconds")
        print(f"Device: {'GPU' if self.use_gpu else 'CPU'}")
        
        return results
    
    def predict(self, x):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Convert to array if DataFrame
        if isinstance(x, pd.DataFrame):
            x_values = x.values
        else:
            x_values = x
        
        # Different prediction approach based on model type
        if self.use_gpu and not hasattr(self.model, 'predict'):
            # Native XGBoost model
            dtest = DMatrix(data=x_values, feature_names=self.feature_names)
            preds = self.model.predict(dtest)
            return preds.astype(int)
        else:
            # Scikit-learn API
            return self.model.predict(x_values)
    
    def get_best_params(self):
        """Return the best hyperparameters found during tuning"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return self.best_params 