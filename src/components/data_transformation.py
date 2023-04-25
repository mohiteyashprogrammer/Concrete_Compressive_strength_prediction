import os
import sys
import pandas as pd 
import numpy as np 
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
## Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object


@dataclass
class DataTRansformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTRansformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Initiated")

            numerical_features = ["Cement","Blast_Furnace_Slag","Superplasticizer","Coarse_Aggregate","Age"]

            logging.info("Pipline Initiated")


            #numerical pipline
            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]
        )

            preprocessor = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features)
            ])

            return preprocessor

            logging.info("Pipline Complited")


        except Exception as e:
            logging.info("Error Occured In Data Transormation Stage")
            raise CustomException(e, sys)


    
    def initated_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Train And Test Data Complited")
            logging.info(f"Train DataFrame Head: \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_data.head().to_string()}")

            logging.info("Optaning Preprocessor Object")

            preprocessor_obj = self.get_data_transformation_obj()

            target_colum_name = "Strength"
            drop_colum = [target_colum_name]


            #Saprate Dependent And Indipendent Features line x and y
            input_features_train_data = train_data.drop(drop_colum,axis=1)
            target_features_train_data = train_data[target_colum_name]

            input_features_test_data = test_data.drop(drop_colum,axis=1)
            target_features_test_data = test_data[target_colum_name]

            # Apply Preprocessor object and transform data
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_data)

            logging.info("Applyed Preprocessor on Train and Test Data")

            ## Convert in To numpy array To become fast
            train_array = np.c_[input_features_train_arr,np.array(target_features_train_data)]
            test_array = np.c_[input_features_test_arr,np.array(target_features_test_data)]

            # Calling Save object fro utils to save preprocessor file
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path
            , obj=preprocessor_obj)

            logging.info("Save Preprocessor Pikel File")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Error Occured in Data Transformation Stage")
            raise CustomException(e, sys)



