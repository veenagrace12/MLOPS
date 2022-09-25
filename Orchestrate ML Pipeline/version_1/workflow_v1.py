from typing import Any, Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import mlflow

def load_data(path: str, unwanted_cols: List) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.drop(unwanted_cols, axis=1, inplace=True)
    return data


def get_classes(target_data: pd.Series) -> List[str]:
    return list(target_data.unique())



def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}

#seperating categorical data
def seperating_categorical(data:pd.DataFrame) -> pd.DataFrame:
    sep_categorical = data.select_dtypes(include=['object'])
    return sep_categorical


def get_ordinal(data: pd.DataFrame) -> pd.DataFrame:
    # scaling the categorical features
    ord_enc = OrdinalEncoder()
    ord_enc.fit(data)
    
    return ord_enc


def rescale_catg_data(data: pd.DataFrame, ord_enc: Any) -> pd.DataFrame:    
    # scaling the categorical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    enc_data_rescaled = pd.DataFrame(ord_enc.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return enc_data_rescaled

#seperating numerical data
def seperating_numerical(data:pd.DataFrame) -> pd.DataFrame:
    sep_numerical = data.select_dtypes(include=['int64', 'float64'])
    return sep_numerical

def get_scaler(data: pd.DataFrame) -> pd.DataFrame:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    
    return scaler


def rescale_num_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the dataframe is converted to a numpy ndarray)
    num_data_rescaled = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return num_data_rescaled

def concat_df(data:pd.DataFrame,data1:pd.DataFrame) -> pd.DataFrame:
    concated_df= pd.concat([data,data1], axis=1)
    return  concated_df






def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, estimator: Any, parameters: List) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)

    with mlflow.start_run():        
        clf = GridSearchCV(
            estimator=estimator, 
            param_grid=parameters, 
            scoring='neg_mean_absolute_error',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train, y_train)
        
        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return clf


# Workflow
def main(path: str):

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diamond Price Prediction Exp Tracker")

    # Define Parameters
    TARGET_COL = 'price'
    UNWANTED_COLS = ['x','y','z']
    TEST_DATA_RATIO = 0.2
    DATA_PATH = path

    # Load the Data
    dataframe = load_data(path=DATA_PATH, unwanted_cols=UNWANTED_COLS)

    # Identify Target Variable
    target_data = dataframe[TARGET_COL]
    input_data = dataframe.drop([TARGET_COL], axis=1)

    # Get Unique Classes
    classes = get_classes(target_data=target_data)
    
    # Split the Data into Train and Test
    train_test_dict = split_data(input_=input_data, output_=target_data, test_data_ratio=TEST_DATA_RATIO)

   
# Preprocessing X_train Data

    # 1.Categorical Train Data
    Categorical_train=seperating_categorical(train_test_dict['X_TRAIN'])
    ord_enc = get_ordinal(Categorical_train)
    catg_train = rescale_catg_data(Categorical_train, ord_enc=ord_enc)
    # 2.Numerical Train Data
    Numerical_train=seperating_numerical(train_test_dict['X_TRAIN'])
    scaler = get_scaler(Numerical_train)
    num_train = rescale_num_data(Numerical_train, scaler=scaler)

    X_train_transformed=concat_df(catg_train,num_train)


 # Preprocessing X_test Data

    # 1.Categorical Test Data
    Categorical_test=seperating_categorical(train_test_dict['X_TEST'])
    ord_enc = get_ordinal(Categorical_test)
    catg_test = rescale_catg_data(Categorical_test, ord_enc=ord_enc)
    # 2.Numerical Test Data
    Numerical_test=seperating_numerical(train_test_dict['X_TEST'])
    scaler = get_scaler(Numerical_test)
    num_test = rescale_num_data(Numerical_test, scaler=scaler)
    
    X_test_transformed=concat_df(catg_test,num_test)



    # Model Training
    ESTIMATOR = DecisionTreeRegressor()
    HYPERPARAMETERS = [{'max_depth':np.arange(1, 21),
              'min_samples_leaf': [1, 5, 10, 20, 35, 50, 100]}]
    regressor = find_best_model(X_train_transformed, train_test_dict['Y_TRAIN'], ESTIMATOR, HYPERPARAMETERS)
    print(regressor.best_params_)
    print(regressor.score(X_test_transformed,train_test_dict['Y_TEST']))
    
# Run the main function
main(path='./data/diamonds.csv')        
