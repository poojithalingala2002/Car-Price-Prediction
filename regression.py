import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('regression')
import pickle

class ALGORITHEM:
    def model_training(X_train,X_test,y_train,y_test):
        try:
            reg_model=LinearRegression()
            reg_model.fit(X_train,y_train)
            logger.info(f'Intercept: {reg_model.intercept_}')
            logger.info(f'Coefficients: {reg_model.coef_}')
            logger.info(f'Train Linear Regression r2_score: {r2_score(y_train, reg_model.predict(X_train))}')
            logger.info(f'Train Linear Regression loss: {mean_squared_error(y_train, reg_model.predict(X_train))}')
            logger.info(f'Test Linear Regression r2_score: {r2_score(y_test,reg_model.predict(X_test))}')
            logger.info(f'Test Linear Regression loss: {mean_squared_error(y_test,reg_model.predict(X_test))}')

            with open('reg_model.pkl','wb') as f1:
                pickle.dump(reg_model,f1)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')