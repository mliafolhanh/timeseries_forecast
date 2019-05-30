import datalabframework
from pyspark.sql.functions import col
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window
from datalabframework.spark.utils import unidecode
import pandas as pd
import os
from datalabframework.spark import dataframe
import numpy as np
os.environ['PYSPARK_PYTHON'] = '/opt/conda/bin/python'
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from models import *
import seaborn as sns
import pickle
from elastictools.doctools import DocTools
import json

def clean_data(total_by_date, max_day):
    """
    Convert data from spark to pandas, clean data, create dummy variables of 
    day of week, month of year, day of month.
    
    Parameters
    ----------
    total_by_date: Spark.DataFrame
        Total sales of a category by date
    
    Returns
    -------
    Pandas.DataFrame
        Total sales of a category by date with clean the outliers and dummy variables
    """
    start_date = total_by_date['transaction_date'].min()
    end_date = max_day
    total_by_date = total_by_date.set_index('transaction_date')
    full_date = pd.DataFrame({'date_ts':pd.date_range(start_date, end_date)}).set_index('date_ts')
    full_quantity_total_product = total_by_date.join(full_date, how = 'right').fillna(0)
    full_quantity_total_product.index.freq = 'D'
    return full_quantity_total_product


def try_forecast_on_one_model(data_series, predictor_cols, target_col, model_class, length_test):
    selected_cols = list(set(predictor_cols).union(set(target_col)))
    n_train = data_series.shape[0] - length_test
    train_series = data_series.iloc[:n_train]
    test_series = data_series.iloc[n_train:]
    model = model_class(train_series, predictor_cols, target_col)
    model.fit()
    preds = model.forecast(test_series.index[0], test_series.index[-1])
    preds = preds.round()
    test_error, test_total_error = model.measure_error(test_series, preds)
    return test_error, test_total_error, model

def try_forecast_on_many_models(data_series, all_models, length_test):
    using_predictor_col = [SeasonalNaiveModel, UpgradeSeasonModel, SarimaxModel]
    flat_result = []
    attr_names = ['model', 'start_date', 'end_date', 'mape', 'mad', 'mean_test', 'mean_preds']
    mad_naive = 0
    for model_class in all_models:
        predictor_cols = []
        if (model_class in using_predictor_col):
            predictor_cols = [f'dow_{i}' for i in range(1, 7)] + [f'moy_{i}' for i in range(2, 13)]\
                            + [f'dom_{i}' for i in range(1, 32)]
            predictor_cols = list(set(predictor_cols).intersection(set(data_series.columns)))
        try:
            test_error, test_total_error, model = try_forecast_on_one_model(data_series, predictor_cols, ['daily_quantity'],\
                                                                     model_class, length_test)
            mean_test, mean_preds = test_error['demand'].mean(), test_error['forecast'].mean()
            result = [model.getLabel(), data_series.index[-length_test].isoformat(), data_series.index[-1].isoformat(),\
                      test_total_error['MAPE'], test_total_error['MAD'], mean_test, mean_preds]
            flat_result.append(dict(zip(attr_names, result)))
            if model_class == NaiveModel:
                mad_naive = test_total_error['MAD']
        except:
            continue
    for result in flat_result:
        if mad_naive != 0:
            result.update({'wape': result['mad']/mad_naive})
        else:
            if result['mad'] == 0:
                result.update({'wape': 1})
            else:
                result.update({'wape': 2})
    return flat_result

def perform_cv_process(data_series, all_models, length_test, n_cv, length_cv):
    flat_result = []
    loc = data_series.shape[0]
    for i in range(n_cv):
        train_series = data_series.iloc[:loc]
        result = try_forecast_on_many_models(train_series, all_models, length_test)
        if result:
            flat_result.extend(result)
        loc = loc - length_test
    return flat_result

def select_best_model(train_series, all_models, length_test, n_cv, length_cv):
    max_possible_n_cv = train_series.shape[0]//length_test - 1
    n_cv = min(n_cv, max_possible_n_cv)
    if n_cv == 0:
        pass
        #todo
    flat_result_cv = perform_cv_process(train_series, list(all_models.values()), length_test, n_cv, length_cv)
    pd_flat_result_cv = pd.DataFrame(flat_result_cv)
    avg_mape = pd_flat_result_cv.groupby('model').agg({'mape':'mean'})
    count_mape = pd_flat_result_cv.groupby('model').agg({'mape': 'count'})
    avg_mape = avg_mape[(count_mape == n_cv)['mape']]
    best_model_name = avg_mape['mape'].argmin()
    best_model = all_models[best_model_name]
    if avg_mape.loc['Naive']['mape'] == 0:
        if avg_mape.loc[best_model_name]['mape'] == 0:
            wape_cv = 1
        else:
            wape_cv = 2
    else:
        wape_cv = avg_mape.loc[best_model_name]['mape']/avg_mape.loc['Naive']['mape']
    using_predictor_col = [SeasonalNaiveModel, UpgradeSeasonModel, SarimaxModel]
    predictor_cols = []
    if (best_model in using_predictor_col):
        predictor_cols = [f'dow_{i}' for i in range(1, 7)] + [f'moy_{i}' for i in range(2, 13)]\
                            + [f'dom_{i}' for i in range(1, 32)]
        predictor_cols = list(set(predictor_cols).intersection(set(df_train.columns)))
    model = best_model(train_series, predictor_cols, ['daily_quantity'])
    model.fit()
    return flat_result_cv, avg_mape['mape'].min(), wape_cv, model

def perform_test_process(data_series, all_models, length_test, n_cv, length_cv):
    train_series = data_series[:-length_test]
    test_series = data_series[-length_test:]
    flat_result_cv, mape_cv, wape_cv, model = select_best_model(train_series, all_models, length_test, n_cv, length_cv)
    preds = model.forecast(test_series.index[0], test_series.index[-1])
    preds = preds.round()
    test_error, test_total_error = model.measure_error(test_series, preds)
    mean_test, mean_preds = test_error['demand'].mean(), test_error['forecast'].mean()
    flat_result_test = {'best_model': model.getLabel(), 'mape_test': test_total_error['MAPE'],\
                        'mad_test': test_total_error['MAD'], 'mean_test': mean_test, 'mean_preds': mean_preds, \
                        'mape_cv': mape_cv, 'wape_cv': wape_cv}
    return flat_result_cv, flat_result_test
    
def retrain_and_predict(data_series, start_future_date, end_future_date, all_models, length_test, n_cv, length_cv):
    future_preds = []
    flat_result_cv, mape_cv, wape_cv, model = select_best_model(data_series, all_models, length_test, n_cv, length_cv)
    preds = model.forecast(start_future_date, end_future_date)
    preds = preds.round()
    predict_time = list(preds.index)
    predict_value = preds.values
    for time, value in zip(predict_time, predict_value):
        result = {'predict_time': time, 'predict_value': value, 'best_model': model.getLabel(), 'mape_cv': mape_cv, 'wape_cv': wape_cv}
        future_preds.append(result)
    return future_preds

def all_forecast_process(total_by_date, freq_, max_day, all_models, length_test, n_cv, length_cv, start_future_date, end_future_date):
    
    df_clean = clean_data(total_by_date, max_day)
    df_clean = df_clean['daily_quantity'].reset_index()
    df_group = df_clean.groupby(pd.Grouper(key ='date_ts', freq = freq_))['daily_quantity'].sum()
    df_group = pd.DataFrame(df_group)
    df_group['moy'] = df_group.index.to_series().dt.month
    df_group = pd.get_dummies(df_group, prefix = ['moy'], columns = ['moy'], drop_first = True)
    if (freq_ == 'W-SUN') | (freq_ == 'D'):
        df_group['dom'] = df_group.index.to_series().dt.day
        df_group = pd.get_dummies(df_group, prefix = ['dom'], columns = ['dom'], drop_first = True)
    if (freq_ == 'D'):
        df_group['dow'] = df_group.index.to_series().dt.dayofweek
        df_group = pd.get_dummies(df_group, prefix = ['dow'], columns = ['dow'], drop_first = True)
    data_series = df_group
    flat_result_cv, flat_result_test = perform_test_process(data_series, all_models, length_test, n_cv, length_cv)
    future_preds = retrain_and_predict(data_series, start_future_date, end_future_date, all_models, length_test, n_cv, length_cv)
    for pred in future_preds:
        pred.update({'last_month_error': flat_result_test['mape_test']})
    return flat_result_cv, flat_result_test, future_preds

def adaptive_forecast_process(total_by_date, freq_):
    total_by_date.columns = ['transaction_date', 'daily_quantity']
    max_day = total_by_date['transaction_date'].max()
    current_day = pd.to_datetime('today')
    current_day = current_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    length_test = 15
    n_cv = 3
    start_future_date = current_day + pd.Timedelta('1 days')
    end_future_date = current_day + pd.Timedelta('15 days')
    if freq_ == 'W-SUN':
        current_day = current_day - pd.Timedelta(f'{current_day.dayofweek + 1} days')
        length_test = 1
        start_future_date = current_day + pd.Timedelta('7 days')
        end_fututre_date = current_day + pd.Timedelta('28 days')
    elif freq_ == 'M':
        if current_day.day < 28:
            current_day = current_day - pd.Timedelta(f'{current_day.day} days')
        length_test = 1
        current_month = current_day.month
        end_day_months = pd.date_range(f'{current_day.year}-01-01', f'{current_day.year}-12-01', freq = 'M')
        start_future_date = end_day_months[current_month  % 12  + 1]
        end_future_date = end_day_months[(current_month + 2) % 12 + 1]
    length_cv = length_test
    all_models = {'Average': AverageModel, 'Naive': NaiveModel, 'Drift': DriftModel, 'SeasonalNaive':SeasonalNaiveModel, \
                 'UpgradeSeason':UpgradeSeasonModel, 'StandardArima': StandardArimaModel, 'SES':SimpleExpSmothingModel, \
                 'Holt': HoltModel, 'HoltWinter': HoltWinterModel, 'DecomposeArima': DecomposeArimaModel, 'Prophet': ProphetModel}
    return all_forecast_process(total_by_date, freq_, current_day, all_models, length_test, n_cv, length_cv, start_future_date, end_future_date)