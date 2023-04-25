import os
import sys
import pandas as pd 
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor

from src.utils import save_object
from src.utils import model_evaluation

@dataclass
class ModelTraningConfig:
    traning_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTaning:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()

    
    def initadted_model_traning(self,train_array,test_array):
        try:
            logging.info("Saprate Dependent and Indipendent Features forn data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )


            models = {
            "LinearRegression":LinearRegression(),
            "Ridge":Ridge(),
            "Lesso":Lasso(),
            "Elastic":ElasticNet(),
            "RandomForestRegressor":RandomForestRegressor(random_state=3)
            }

            model_report:dict = model_evaluation(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n*************************************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## to get Best Model Score From Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name:{best_model_name}, R2 Score:{best_model_score}")
            print("\n***************************************************************\n")
            logging.info(f"Best Model Found, Model Name:{best_model_name}, R2 Score:{best_model_score}")

            save_object(file_path = self.model_trainer_config.traning_model_file_path,
             obj=best_model)

        except Exception as e:
            logging.info("Error Occured Model Traning")
            raise CustomException(e, sys)