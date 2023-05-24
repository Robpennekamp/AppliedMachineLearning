import itertools
import pickle
from datetime import date
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time


def read_model(dat):
    """
    Function used to read in either a string or a datetime.date() format date. 
    
    Will return the pickle of the best model generated on that day.
    Format:
    List:
    - Model
    - RMSE achieved
    - Model name
    """
    if type(dat) is str:
        try:
            dat = datetime.strptime(dat, '%Y-%m-%d').date()
        except:
            print("Can't convert date, please retry! The format should either be a string: 2023-05-17. Or datetime.date type.")
            return None
    x = pickle.load(open(f'best_model_{dat}.pkl', 'rb'))
    return x


def find_optimal_parameters(data, model, params, n_splits=5, verbose=False, pickle_best=False, interval='1H'):
    """
    Function which will find the optimal parameters within set boudaries.
    Receives as input:
    - The data
    - A model
    - The dictionary of parameters to optimize:
        - Key: parameter name
        - Value: List of testing parameter values.
    - The amount of Timeseries splits for crossvalidation (Standard: 5)
    - Whether information should be printed or not (Standard: False)
    - Whether the model with the best achieved RMSE should still be saved
    
    Output:
    A dictionary consisting of:
    - Key: Model_Name
    - Values: List of:
        - Model parameters:
        - Result list:
            - model
            - Achieved RMSE
            - List of:
                - Training Errors
                - Testing Errors
            - Time taken to fit fold.
    """
    
    parameter_combinations = itertools.product(*params.values())
    results = {}
    count = 1
    for x in parameter_combinations:
        model_name = f'Model {count}'
        results[model_name] = [dict(zip(params.keys(), x))]
        count +=1
    lowest = 1000000000
    best_model = ""
    for model_name in results.keys():
        params_dict = results[model_name][0]
        print(f'Testing model {model_name} with parameters: {params_dict}')
        current_model = model(**params_dict)
        results[model_name].append(run_timeseries_model(data, current_model, n_splits=n_splits, verbose=verbose, interval=interval))
        total_time = sum(results[model_name][1][3])
        print(f'Finished testing model. Test RMSE: {results[model_name][1][1]}. Time taken: {total_time}.')
        if results[model_name][1][1] < lowest:
            lowest = results[model_name][1][1]
            best_model = results[model_name][1][0]
            best_model_name = model_name
    print(f"""Optimization completed. Best model is {best_model} with parameters \n\n {results[best_model_name]}. \n
    RMSE is {results[best_model_name][1][1]}.""")
    try: 
        cur_best = pickle.load(open(f'best_model_{date.today()}.pkl', 'rb'))[1]
        if cur_best > lowest:
            print("New best rmse reached! Pickling model.")
            pickle.dump([best_model, lowest, best_model_name], open(f"Best_model_{date.today()}.pkl", 'wb'))
        elif pickle_best:
            print("RMSE of best model has not improved. Saving model anyways.")
            pickle.dump([best_model, lowest, best_model_name], open(f'Saved_model_{datetime.now()}.pkl', 'wb'))
        else:
            print("RMSE of best model does not improve on the best model of today. Not saving model.")
    except FileNotFoundError:
        print("New best rmse reached! Pickling model.")
        pickle.dump([best_model, lowest, best_model_name], open(f"Best_model_{date.today()}.pkl", 'wb'))
    
    return results


def run_timeseries_model(data, model, n_splits = 5, verbose=False, interval='1H'):
    """
    Function which receives a data and a model as input.
    Can also receive the amount of splits you want as input.
    For debugging purposes, set verbose to True.
    The interval in which the training data must be resampled.
    
    Outputs a list of:
        - The resulting model
        - The final RMSE
        - A list of the resulting Train RMSEs (per fold)
        - A list of the resulting Test RMSEs (per fold)
        - A list of the time it took to train over each fold.
    """
    
    ts = TimeSeriesSplit(n_splits=n_splits)
    count = 1
    train_rmses = []
    test_rmses = []
    fold_training_times = []
    for train, test in ts.split(data):
        if verbose:
            print(f'Starting fold {count}')
        cv_train, cv_test = data.iloc[train], data.iloc[test]
        cv_train = cv_train.resample(interval).last()
        y_train = cv_train['demand_kW']
        x_train = cv_train.drop(['demand_kW'], axis=1)
        y_test = cv_test['demand_kW']
        x_test = cv_test.drop(['demand_kW'], axis=1)
        train_start = time.time()
        if verbose:
            print("Training model...")
        model.fit(x_train, y_train)
        
        if verbose:
            print("Predicting...")
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        
        if verbose:
            print("Calculating rmse's...")
        train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        rmse_stop = time.time()
        
        train_rmses.append(train_rmse)
        test_rmses.append(test_rmse)
        fold_training_times.append(rmse_stop - train_start)
        if verbose:
            print(f'Fold {count} train error: {train_rmse}. Test error: {test_rmse}. Time taken: {rmse_stop - train_start} s.')
        count += 1
    all_rmses = [train_rmses, test_rmses]
    return [model, test_rmse, all_rmses, fold_training_times]