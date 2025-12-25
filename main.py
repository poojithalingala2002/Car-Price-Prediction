import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('main')
from missing_value_handling import MISSING_VAL
from variable_trans_out_handle import VAR_TRANS_OUT_HANDLE
from cat_to_num import CATEGORYTONUM
from feature_select import FEATURE
from data_scaling import DATASCALE
from regression import ALGORITHEM


class CAR_PRICE_PREDICTION:
    def __init__(self,path):
        try:
            self.path=path
            self.df=pd.read_csv(self.path)
            logger.info(f'shape of df:{self.df.shape}')
            logger.info(f'{self.df.columns}')
            logger.info(f'{self.df.isnull().sum()}')
            self.y = self.df['Selling_Price']
            self.X = self.df.drop(['Selling_Price'], axis=1)
            logger.info(f'shape of X:{self.X.shape}')
            logger.info(f'shape of y:{self.y.shape}')
            logger.info(f'{self.X.columns}')
            logger.info(f'{self.y.name}')
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'shape of X_train:{self.X_train.shape}')
            logger.info(f'shape of X_test:{self.X_test.shape}')
            logger.info(f'shape of y_train:{self.y_train.shape}')
            logger.info(f'shape of y_test:{self.y_test.shape}')
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            logger.info(f'columns of y_train:{self.y_train.name}')
            logger.info(f'columns of y_test:{self.y_test.name}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def missing_values(self):
        try:
            if self.X_train.isnull().sum().all() > 0 or self.X_test.isnull().sum().all() > 0:
                self.X_train,self.X_test=MISSING_VAL.random_sample(self.X_train,self.X_test)
            else:
                logger.info(f'There are no missing values in X_train and X_test')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def feature_engineering(self):
        try:
            logger.info('Starting feature engineering (Brand extraction)')
            def extract_brand(car_name):
                if pd.isna(car_name):
                    return 'Unknown'
                name = car_name.strip().lower()
                multi_word_brands = [
                    'royal enfield',
                    'hero honda'
                ]
                for brand in multi_word_brands:
                    if name.startswith(brand):
                        return brand.title()
                return name.split()[0].title()
            # Create Brand column
            self.X_train['Brand'] = self.X_train['Car_Name'].apply(extract_brand)
            self.X_test['Brand'] = self.X_test['Car_Name'].apply(extract_brand)
            # Drop high-cardinality column
            self.X_train.drop(['Car_Name'], axis=1, inplace=True)
            self.X_test.drop(['Car_Name'], axis=1, inplace=True)
            logger.info('Brand extraction completed')
            logger.info(f'X_train columns after feature engineering: {self.X_train.columns}')
            logger.info(f'X_test columns after feature engineering: {self.X_test.columns}')
            logger.info(f'\n{self.X_train.head(10)}')
            logger.info(f'\n{self.X_test}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')


    def vt_outhand(self):
        try:
            #logger.info(f'{self.X_train.info()}')
            logger.info(f'columns of X_train:{self.X_train.columns}')
            logger.info(f'columns of X_test:{self.X_test.columns}')
            self.X_train_num=self.X_train.select_dtypes(exclude=object)
            self.X_train_cat=self.X_train.select_dtypes(include=object)
            self.X_test_num=self.X_test.select_dtypes(exclude=object)
            self.X_test_cat=self.X_test.select_dtypes(include=object)
            logger.info(f'columns of X_train_num:{self.X_train_num.columns}')
            logger.info(f'columns of X_train_cat:{self.X_train_cat.columns}')
            logger.info(f'columns of X_test_num:{self.X_test_num.columns}')
            logger.info(f'columns of X_test_cat:{self.X_test_cat.columns}')
            # logger.info(f'shape of X_train_num:{self.X_train_num.shape}')
            # logger.info(f'shape of X_train_cat:{self.X_train_cat.shape}')
            # logger.info(f'shape of X_test_num:{self.X_test_num.shape}')
            # logger.info(f'shape of X_test_cat:{self.X_test_cat.shape}')
            #logger.info(f'\n{self.X_train_num.head(10)}')
            self.X_train_num,self.X_test_num=VAR_TRANS_OUT_HANDLE.variable_transform_outliers(self.X_train_num,self.X_test_num)
            logger.info(f'===========================================================')
            logger.info(f'columns of X_train_num:{self.X_train_num.columns}')
            logger.info(f'columns of X_train_cat:{self.X_train_cat.columns}')
            logger.info(f'columns of X_test_num:{self.X_test_num.columns}')
            logger.info(f'columns of X_test_cat:{self.X_test_cat.columns}')
            # logger.info(f'shape of X_train_num:{self.X_train_num.shape}')
            # logger.info(f'shape of X_train_cat:{self.X_train_cat.shape}')
            # logger.info(f'shape of X_test_num:{self.X_test_num.shape}')
            # logger.info(f'shape of X_test_cat:{self.X_test_cat.shape}')
            #logger.info(f'\n{self.X_train_num.head(10)}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def categori_num(self):
        try:
            # for i in self.X_train_cat.columns:
            #     logger.info(f'\n{self.X_train_cat[i].unique()}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            self.X_train_cat,self.X_test_cat=CATEGORYTONUM.category_to_numeric(self.X_train_cat,self.X_test_cat,self.y_train)
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f"{self.X_train_cat.shape}")
            logger.info(f"{self.X_test_cat.shape}")
            logger.info(f"{self.X_train_cat.isnull().sum()}")
            logger.info(f"{self.X_test_cat.isnull().sum()}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def feature_select(self):
        try:
            logger.info(f'Before :{self.X_train_num.columns}-->{self.X_train_num.shape}')
            logger.info(f'Before :{self.X_test_num.columns}-->{self.X_test_num.shape}')
            self.X_train_num,self.X_test_num=FEATURE.feature_selecting(self.X_train_num,self.X_test_num,self.y_train)
            logger.info(f'After :{self.X_train_num.columns}-->{self.X_train_num.shape}')
            logger.info(f'After :{self.X_test_num.columns}-->{self.X_test_num.shape}')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def feature_scaling(self):
        try:
            #logger.info(f'{self.y_train}')
            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)
            logger.info(f'Before \n:{self.training_data}')
            logger.info(f'Before \n:{self.testing_data}')
            self.training_data,self.testing_data=DATASCALE.data_scale(self.training_data,self.testing_data)
            logger.info(f'After \n:{self.training_data}')
            logger.info(f'After \n:{self.testing_data}')
            logger.info(f'================================================')
            logger.info(f"{self.training_data.isnull().sum()}")
            logger.info(f"{self.testing_data.isnull().sum()}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

    def all_algo(self):
        try:
            logger.info(f'========================================================')
            logger.info(f'\n{self.training_data}')
            logger.info(f'\n{self.testing_data}')
            logger.info(f'\n{self.y_train}')
            logger.info(f'\n{self.y_test}')
            ALGORITHEM.model_training(self.training_data,self.testing_data,self.y_train,self.y_test)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')

if __name__ == '__main__':
    try:
        obj=CAR_PRICE_PREDICTION('D:\\Projects\\Car_prize_prediction\\data.csv')
        obj.missing_values()
        obj.feature_engineering()
        obj.vt_outhand()
        obj.categori_num()
        #obj.feature_select()
        obj.feature_scaling()
        obj.all_algo()
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in line no:{error_line.tb_lineno} due to:{error_msg}')
