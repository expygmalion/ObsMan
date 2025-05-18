from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class RandomForestModel:
    def __init__(self, random_state=2021):
        self.random_state = random_state
        self.model = None
        
    def train(self, x_train, y_train):
        """Train Random Forest model with GridSearchCV for hyperparameter tuning"""
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        self.model = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state), 
            param_grid, 
            cv=5
        )
        
        self.model.fit(x_train, y_train)
        return self.model
    
    def evaluate(self, x_train, y_train, x_test, y_test):
        """Evaluate the model on training and test data"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get predictions
        train_preds = self.model.predict(x_train)
        test_preds = self.model.predict(x_test)
        
        # Calculate metrics
        results = {}
        
        # Accuracy
        results['train_accuracy'] = accuracy_score(y_train, train_preds)
        results['test_accuracy'] = accuracy_score(y_test, test_preds)
        
        # Precision
        results['train_precision'] = precision_score(y_train, train_preds, average='weighted')
        results['test_precision'] = precision_score(y_test, test_preds, average='weighted')
        
        # Recall
        results['train_recall'] = recall_score(y_train, train_preds, average='weighted')
        results['test_recall'] = recall_score(y_test, test_preds, average='weighted')
        
        # F1 Score
        results['train_f1'] = f1_score(y_train, train_preds, average='weighted')
        results['test_f1'] = f1_score(y_test, test_preds, average='weighted')
        
        # Confusion Matrix
        results['test_confusion_matrix'] = confusion_matrix(y_test, test_preds)
        
        return results
    
    def predict(self, x):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return self.model.predict(x)
    
    def get_best_params(self):
        """Return the best hyperparameters found during tuning"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return self.model.best_params_
    
    def get_feature_importance(self):
        """Return feature importance rankings from the model"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return self.model.best_estimator_.feature_importances_ 