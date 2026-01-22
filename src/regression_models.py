from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

class RegressionModel:
    """
    Base class for regression models.
    """
    
    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
    def create_model(self):
        """Create the regression model based on type."""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        if self.model is None:
            self.create_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"{self.model_type.capitalize()} model trained successfully")
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Features to predict
            
        Returns:
            np.array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
