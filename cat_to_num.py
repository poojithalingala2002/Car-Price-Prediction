import sys
from unittest import result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
import pickle
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('cat_to_num')

class CATEGORYTONUM:
    def category_to_numeric(X_train,X_test,y_train):
        try:

            logger.info(f'{X_train.columns}')
            for i in X_train.columns:
                logger.info(f'\n{i}--->{X_train[i].unique()}')

            X_train['Fuel_Type']=X_train['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})
            X_train['Seller_Type']=X_train['Seller_Type'].map({'Individual':0,'Dealer':1})
            X_train['Transmission']=X_train['Transmission'].map({'Manual':0,'Automatic':1})

            X_test['Fuel_Type']=X_test['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
            X_test['Seller_Type']=X_test['Seller_Type'].map({'Individual': 0, 'Dealer': 1})
            X_test['Transmission']=X_test['Transmission'].map({'Manual': 0, 'Automatic': 1})

            te=TargetEncoder()
            te.fit(X_train[['Brand']],y_train)
            X_train['Brand']=te.transform(X_train[['Brand']])
            logger.info(f'\n{X_train}')

            X_test['Brand']=te.transform(X_test[['Brand']])
            logger.info(f'\n{X_test}')

            with open('target_encode.pkl','wb') as f2:
                pickle.dump(te,f2)
            return X_train,X_test

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')