from credit.entity import artifact_entity,config_entity
from credit.exception import CreditException
from credit.logger import logging
from typing import Optional
import os,sys 
from sklearn.pipeline import Pipeline
import pandas as pd
from credit import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from credit.config import TARGET_COLUMN
from collections import Counter



class DataTransformation:


    def __init__(self,data_transformation_config:config_entity.DataTransformationConfig,
                    data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config=data_transformation_config
            self.data_ingestion_artifact=data_ingestion_artifact
        except Exception as e:
            raise CreditException(e, sys)

    
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            pass
        except Exception as e:
            raise CreditException(e, sys)
            


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            #reading training and testing file
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            #selecting input feature for train and test dataframe
            input_feature_train_df=train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df=test_df.drop(TARGET_COLUMN,axis=1)

            #selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            
            '''
            transformation_pipleine = DataTransformation.get_data_transformer_object()
            transformation_pipleine.fit(input_feature_train_df)
            '''
    
            '''
            #transforming input features
            input_feature_train_arr = transformation_pipleine.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipleine.transform(input_feature_test_df)
            '''
            
            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in training set Input: {input_feature_train_df.shape} Target:{target_feature_train_df.shape}")
            input_feature_train_df, target_feature_train_df = smt.fit_resample(input_feature_train_df, target_feature_train_df)
            logging.info(f"After resampling in training set Input: {input_feature_train_df.shape} Target:{target_feature_train_df.shape}")
            
            logging.info(f"Before resampling in testing set Input: {input_feature_test_df.shape} Target:{target_feature_test_df.shape}")
            input_feature_test_df, target_feature_test_df = smt.fit_resample(input_feature_test_df, target_feature_test_df)
            logging.info(f"After resampling in testing set Input: {input_feature_test_df.shape} Target:{target_feature_test_df.shape}")
            logging.info(f"The No. of 0's & 1's in training set Input: {Counter(target_feature_train_df)} The No. of 0's & 1's in testing set Input:{Counter(target_feature_test_df)}")
            
            
            #target encoder
            train_arr = np.c_[input_feature_train_df, target_feature_train_df]
            test_arr = np.c_[input_feature_test_df, target_feature_test_df]


            #save numpy array
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_path,
                                        array=train_arr)

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_path,
                                        array=test_arr)

            

            '''
            utils.save_object(file_path=self.data_transformation_config.transform_object_path,
             obj=transformation_pipleine)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path,
            obj=label_encoder)
            '''
           

            
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                #transform_object_path=self.data_transformation_config.transform_object_path,
                transformed_train_path = self.data_transformation_config.transformed_train_path,
                transformed_test_path = self.data_transformation_config.transformed_test_path,
                #target_encoder_path = self.data_transformation_config.target_encoder_path

            )
            

            logging.info(f"Data transformation object {data_transformation_artifact}")
            return data_transformation_artifact
            
        except Exception as e:
            raise CreditException(e, sys)