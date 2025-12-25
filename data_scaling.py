import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('data_scaling')
import pickle
from sklearn.preprocessing import StandardScaler

class DATASCALE:
    def data_scale(X_train,X_test):
        try:
            logger.info(f'\n{X_train}')
            logger.info(f'\n{X_test}')

            scale_cols=['Year','Present_Price','Kms_Driven','Brand']
            sc=StandardScaler()
            sc.fit(X_train[scale_cols])

            X_train[scale_cols]=sc.transform(X_train[scale_cols])
            X_test[scale_cols] = sc.transform(X_test[scale_cols])

            with open('scaler.pkl','wb') as f:
                pickle.dump(sc,f)
            return X_train,X_test
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')