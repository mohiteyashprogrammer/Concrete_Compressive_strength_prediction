import os
import sys
import pandas as pd 
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pickle

from src.utils import load_object


@dataclass
class PredictPipline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            logging.info("Prediction Pipline Started")
            # This Line of code eork in any system 
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            #Load Object
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info("Error Occured in Prediction Pipline")
            raise CustomException(e, sys)



# Create Custom Data Class
class CustomData:
    def __init__(self,
        Cement:float,
        Blast_Furnace_Slag:float,
        Superplasticizer:float,
        Coarse_Aggregate:float,
        Age:float):


        self.Cement = Cement
        self.Blast_Furnace_Slag = Blast_Furnace_Slag
        self.Superplasticizer = Superplasticizer
        self.Coarse_Aggregate = Coarse_Aggregate
        self.Age = Age


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict= {
                "Cement":[self.Cement],
                "Blast_Furnace_Slag":[self.Blast_Furnace_Slag],
                "Superplasticizer":[self.Superplasticizer],
                "Coarse_Aggregate":[self.Coarse_Aggregate],
                "Age":[self.Age]

            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured  In Prediction Pipline")
            raise CustomException(e, sys)




