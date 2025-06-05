import os
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves the given object to the specified file path using pickle.
    
    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The object to be saved.
    
    Returns:
    None
    """
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params, cv=3, n_jobs=3, verbose=1):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param=params[list(models.keys())[i]]

            gs = GridSearchCV(estimator=model, param_grid=param, cv=cv, n_jobs=n_jobs, verbose=verbose)
            gs.fit(X_train, y_train)  # Fit the model with GridSearchCV

            ##model.fit(X_train, y_train)  # Train the model
            model.set_params(**gs.best_params_) # Set the best parameters found by GridSearchCV
            model.fit(X_train, y_train) # Set the best parameters found by GridSearchCV
            logging.info(f"Best parameters for {list(models.keys())[i]}: {gs.best_params_}")


            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                'train_score': train_model_score,
                'test_score': test_model_score,
                'model': model
            }

        logging.info(f"Model evaluation report: {report}")

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)
