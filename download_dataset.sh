#!/bin/bash
curl -L -o ./dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/nudratabbas/vitamin-deficiency-disease-prediction-dataset
unzip dataset.zip -d ./dataset
rm dataset.zip
mv ./dataset/vitamin_deficiency_disease_dataset_20260123.csv ./dataset/data.csv