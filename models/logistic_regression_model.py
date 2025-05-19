from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class LogisticRegressionModel:
    def __init__(self, random_state=2021, poly_degree=2, max_iter=1000):
        self.random_state = random_state
        self.poly_degree = poly_degree
        self.max_iter = max_iter
        self.model = None
        self.poly = None
        
    def train(self, x_train, y_train):
        """Train Logistic Regression model with polynomial features"""
        # Generate polynomial features
        self.poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        x_train_poly = self.poly.fit_transform(x_train)
        
        # Initialize and train the model
        self.model = LogisticRegression(random_state=self.random_state, max_iter=self.max_iter)
        self.model.fit(x_train_poly, y_train)
        
        return self.model
    
    def evaluate(self, x_train, y_train, x_test, y_test):
        """Evaluate the model on training and test data"""
        if self.model is None or self.poly is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Transform data with polynomial features
        x_train_poly = self.poly.transform(x_train)
        x_test_poly = self.poly.transform(x_test)
        
        # Get predictions
        train_preds = self.model.predict(x_train_poly)
        test_preds = self.model.predict(x_test_poly)
        
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
        if self.model is None or self.poly is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        x_poly = self.poly.transform(x)
        return self.model.predict(x_poly)
    
    def get_model_coef(self):
        """Return the model coefficients"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return self.model.coef_ 