import pymongo
from credit.exception import CreditException
from credit.logger import logging
from credit.pipeline.training_pipeline import start_training_pipeline

file_path="/config/workspace/processed_South_German_Bank_Credit_Risk_dataset.csv"
print(__name__)
if __name__=="__main__":
     try:

          

          #data ingestion
          start_training_pipeline()

          '''
          start_training_pipeline()
          output_file = start_batch_prediction(input_file_path=file_path)
          print(output_file)
          '''
     except Exception as e:
          print(e)