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

def get_ordinal(data: pd.DataFrame) -> Any:
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

def get_scaler(data: pd.DataFrame) -> Any:
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







def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}


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

   


    # Rescaling Train and Test Categorical Data
    ord_enc = get_ordinal(train_test_dict['X_TRAIN'].select_dtypes(include=['object']))
    catg_train = rescale_catg_data(data=train_test_dict['X_TRAIN'].select_dtypes(include=['object']), ord_enc=ord_enc)
    catg_test = rescale_catg_data(data=train_test_dict['X_TEST'].select_dtypes(include=['object']), ord_enc=ord_enc)
    
    
    catg_train = pd.DataFrame(catg_train)
    catg_test = pd.DataFrame(catg_test)

    # Rescaling Train and Test Numerical Data
    scaler = get_scaler(train_test_dict['X_TRAIN'].select_dtypes(exclude=['object']))
    num_train = rescale_num_data(data=train_test_dict['X_TRAIN'].select_dtypes(exclude=['object']), scaler=scaler)
    num_test = rescale_num_data(data=train_test_dict['X_TEST'].select_dtypes(exclude=['object']), scaler=scaler)

    
    num_train = pd.DataFrame(num_train)
    num_test = pd.DataFrame(num_test)

    # Rescaling Train and Test Data

    X_train_transformed = pd.concat([num_train,catg_train], axis=1)
    X_test_transformed = pd.concat([num_test,catg_test], axis=1)



    # Model Training
    ESTIMATOR = DecisionTreeRegressor()
    HYPERPARAMETERS = [{'max_depth':np.arange(1, 21),
              'min_samples_leaf': [1, 5, 10, 20, 35, 50, 100]}]
    regressor = find_best_model(X_train_transformed, train_test_dict['Y_TRAIN'], ESTIMATOR, HYPERPARAMETERS)
    print(regressor.best_params_)
    
    
# Run the main function
main(path='./data/diamonds.csv')        
