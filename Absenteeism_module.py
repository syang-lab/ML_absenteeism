
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class absenteeism_model():
    """This model is used to predict absenteeism of new data"""
    def __init__(self,model_file):
        with open('absenteeism_model','rb') as file:
            self.classifier = pickle.load(file)
            self.data = None
    
    def data_preprocess(self, data_file):
        raw_data = pd.read_csv(data_file)
        data_preprocessed = raw_data.drop(['ID'],axis = 1)
        # encode the categorical data # group similar absence reasons together
        reason_columns = pd.get_dummies(data_preprocessed['Reason for Absence'], drop_first = True)
        reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
        
        data_preprocessed = pd.concat([reason_type_1,reason_type_2,reason_type_3,reason_type_4,data_preprocessed], axis = 1)
        # reason1: various diseases, reason2: pregnancy reason3: poisoning reason4: light diseases
        column_names = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Reason for Absence', 'Date', 'Transportation Expense','Distance to Work', 'Age', 'Daily Work Load Average','Body Mass Index', 'Education', 'Children', 'Pets','Absenteeism Time in Hours']
        data_preprocessed.columns = column_names
        data_preprocessed = data_preprocessed.drop(['Reason for Absence','Date'], axis =1)
        
        data_preprocessed.to_csv('data_preprocessed')
        self.data = data_preprocessed
        
    def scaled_feature(self, data_file):
        self.x= self.data.iloc[:,:].values
        mms = MinMaxScaler()
        self.x = mms.fit_transform(self.x)
    
    def predict_outputs(self):
        if (self.data is not None):
            self.y_pred = self.classifier.predict(self.x)
        self.data['prediction'] = self.y_pred
        return self.y_pred;        

