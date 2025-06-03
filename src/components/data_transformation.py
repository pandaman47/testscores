import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This method creates a preprocessor object that applies transformations to the dataset.
        '''
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created successfully")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_object()
            #print(preprocessor_obj)

            target_column = 'math_score'
            numerical_features = ['writing_score', 'reading_score']
            input_features_train = train_df.drop(columns=[target_column], axis=1)
            input_features_test = test_df.drop(columns=[target_column], axis=1)
            target_feature_train = train_df[target_column]
            target_feature_test = test_df[target_column]

            logging.info("Applying transformations to training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_features_train)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            logging.info("saved preprocessor object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)