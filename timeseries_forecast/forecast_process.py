from models import *


def clean_data(total_by_date, last_day):
    """
    Returns
    -------
    Pandas.DataFrame
        Total sales of a category by date with clean the outliers and dummy variables
    """
    start_date = total_by_date['transaction_date'].min()
    end_date = last_day
    total_by_date = total_by_date.set_index('transaction_date')
    full_date = pd.DataFrame({'transaction_date':pd.date_range(start_date, end_date)}).set_index('transaction_date')
    full_quantity_total_product = total_by_date.join(full_date, how = 'right').fillna(0)
    full_quantity_total_product.index.freq = 'D'
    return full_quantity_total_product


def try_forecast_on_one_model(data_series, predictor_cols, target_col, model_class, length_test, flag = None):
    """
    Try forecast on data_series using model_class
    
    Parameters
    ----------
    data_series: Spark.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    predictor_cols: List
        List of predictor columns
    target_col: List
        The column include total sale information
    model_class: TimeseriesForecastingModel
        Model using for forecasting
    length_test: int
        The length of test part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
    Returns
    -------
    Pandas.DataFrame:
        Detail information of forecasting result
    Dict:
        Total error of forecasting result: MSE, MAPE, MAD
    TimeseriesForecastingModel:
        Model after fiting the time series
    """
    selected_cols = list(set(predictor_cols).union(set(target_col)))
    n_train = data_series.shape[0] - length_test
    train_series = data_series.iloc[:n_train]
    test_series = data_series.iloc[n_train:]
    model = model_class(train_series, predictor_cols, target_col)
    model.fit()
    preds = model.forecast(test_series.index[0], test_series.index[-1])
    preds = preds.round()
    test_error, test_total_error = model.measure_error_2(test_series, preds, flag = flag)
    return test_error, test_total_error, model


def cal_wape_value(naive_value, model_value):
    """
    Function for calculating the WAPE value
    
    Parameters
    ----------
    nanive_value: Float
        The value of Naive method
    model_value: Float
        The value of model 
    
    Returns
    -------
    Float:
        WAPE value
    """
    if naive_value == 0:
        if model_value == 0:
            wape = 1
        else:
            wape = 2
    else:
        wape = model_value/naive_value
    return wape


def try_forecast_on_many_models(data_series, all_models, length_test, flag = None):
    """
    Try forecasting on time series by using many models
    
    Parameters
    ----------
    data_series: Spark.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    all_models: List of TimeseriesForecastingModel
        All model classes using for forecasting
    length_test: int
        The length of test part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
    Returns
    -------
    List of dict:
        Each dict present information of result when using a given forecasting model. Includes:
        model: Name of model, start_date: Start date of test part, end_date: End date of test part
        mape: Mean absolute percent error, mad: Mean absolute deviation, mean_test: Mean of actual value of test part
        mean_preds: Mean of forecasting value
    """
    using_predictor_col = [SeasonalNaiveModel, UpgradeSeasonModel, SarimaxModel]
    flat_result = []
    attr_names = ['model', 'start_date', 'end_date', 'mape', 'mad', 'mean_test', 'mean_preds']
    mad_naive = 0
    for model_class in all_models:
        predictor_cols = []
        if (model_class in using_predictor_col):
            predictor_cols = [f'dow_{i}' for i in range(1, 7)] + [f'moy_{i}' for i in range(2, 13)]\
                            + [f'dom_{i}' for i in range(1, 32)]
            predictor_cols = list(set(predictor_cols) & set(data_series.columns))
        try:
            test_error, test_total_error, model = try_forecast_on_one_model(data_series, predictor_cols, ['daily_quantity'],
                                                                            model_class, length_test, flag = flag)
            mean_test, mean_preds = test_error['demand'].mean(), test_error['forecast'].mean()
            result = [model.getLabel(), data_series.index[-length_test].isoformat(), data_series.index[-1].isoformat(),\
                      test_total_error['MAPE'], test_total_error['MAD'], mean_test, mean_preds]
            flat_result.append(dict(zip(attr_names, result)))
            if model_class == NaiveModel:
                mad_naive = test_total_error['MAD']
        except:
            continue
    for result in flat_result:
        wape = cal_wape_value(mad_naive, result['mad'])
        result.update({'wape': wape})
    return flat_result


def perform_cv_process(data_series, all_models, n_cv, length_cv, flag = None):
    """
    Perform cross validation process on a time series
    
    Parameters
    ----------
    data_series: Spark.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    all_models: List of TimeseriesForecastingModel
        All model classes using for forecasting
    n_cv: int
        Number folds of cross validation
    length_cv: int
        The length of cv part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
        
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part
    """
    flat_result = []
    loc = data_series.shape[0]
    for i in range(n_cv):
        train_series = data_series.iloc[:loc]
        result = try_forecast_on_many_models(train_series, all_models, length_cv, flag = flag)
        if result:
            flat_result.extend(result)
        loc = loc - length_cv
    return flat_result


def select_best_model(train_series, all_models, n_cv, length_cv, flag = None):
    """
    Function for selecting the best model using cross validation
    
    Parameters
    ----------
    train_series: Spark.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    all_models: dict of TimeseriesForecastingModel
        All model classes using for forecasting
    n_cv: int
        Number folds of cross validation
    length_cv: int
        The length of cv part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
            
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part
    Float:
        The best MAPE of best model
    Float:
        The WAPE of best model on cv part
    Float:
        The standard deviation of erros of best model on CV part
    TimeseriesForecastingModel:
        The best model after fiting
    """
    max_possible_n_cv = train_series.shape[0]//length_cv - 1
    n_cv = min(n_cv, max_possible_n_cv)
    if n_cv <= 0:
        return None, None, None, None, None
    flat_result_cv = perform_cv_process(train_series, list(all_models.values()), n_cv, length_cv, flag = flag)
    pd_flat_result_cv = pd.DataFrame(flat_result_cv)
    model_info = pd_flat_result_cv.groupby('model').agg({'mape': ['count', 'mean', 'std']})
    model_info.columns = ['count', 'mean', 'std']
    naive_mape = model_info.loc['Naive']['mean']
    model_info = model_info[model_info['count'] == n_cv].reset_index().values
    model_info = sorted(model_info, key = lambda x:x[2]+x[3])
    using_predictor_col = [SeasonalNaiveModel, UpgradeSeasonModel, SarimaxModel]
    for one_model in model_info:
        wape_cv = cal_wape_value(naive_mape, one_model[2])
        try:
            model_class = all_models[one_model[0]]
            predictor_cols = []
            if model_class in using_predictor_col:
                predictor_cols = [f'dow_{i}' for i in range(1, 7)] + [f'moy_{i}' for i in range(2, 13)]\
                                    + [f'dom_{i}' for i in range(1, 32)]
                predictor_cols = list(set(predictor_cols) & set(train_series.columns))
            model = model_class(train_series, predictor_cols, ['daily_quantity'])
            model.fit()
            return flat_result_cv, one_model[2], wape_cv, one_model[3], model
        except:
            continue
    return None, None, None, None, None


def perform_test_process(data_series, all_models, length_test, n_cv, length_cv, flag = None):
    """
    Function with evaluation the generalization error
    
    Parameters
    ----------
    data_series: pandas.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    all_models: List of TimeseriesForecastingModel
        All model classes using for forecasting
    length_test: int
        The length of test part
    n_cv: int
        Number folds of cross validation
    length_cv: int
        The length of cv part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
        
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part
    Dict:
        All information of forecasting result for test part with using the best model which get from CV process
    """
    train_series = data_series[:-length_test]
    test_series = data_series[-length_test:]
    flat_result_cv, mape_cv, wape_cv, std_cv, model = select_best_model(train_series, all_models, n_cv, length_cv, flag = flag)
    if model is None:
        return None, None
    preds = model.forecast(test_series.index[0], test_series.index[-1])
    preds = preds.round()
    test_error, test_total_error = model.measure_error_2(test_series, preds, flag = flag)
    mean_test, mean_preds = test_error['demand'].mean(), test_error['forecast'].mean()
    naivemodel = NaiveModel(train_series, [], ['daily_quantity'])
    naivemodel.fit()
    naivepreds = naivemodel.forecast(test_series.index[0], test_series.index[-1])
    naivepreds = naivepreds.round()
    naive_test_error, naive_test_total_error = naivemodel.measure_error_2(test_series, naivepreds, flag = flag)
    wape_test = cal_wape_value(naive_test_total_error['MAPE'], test_total_error['MAPE'])
    flat_result_test = {'best_model': model.getLabel(), 'mape_test': test_total_error['MAPE'],
                        'wape_test': wape_test, 'mad_test': test_total_error['MAD'],
                        'mean_test': mean_test, 'mean_preds': mean_preds, 'mape_cv': mape_cv, 'wape_cv': wape_cv}
    return flat_result_cv, flat_result_test


def retrain_and_predict(data_series, start_future_date, end_future_date, all_models, n_cv, length_cv, flag=None):
    """
    The procedure for retrain and predict future data
    
    Parameters
    ----------
    data_series: Spark.DataFrame
        Dataframe includes information about sale and all necessary predictor columns
    start_future_date: Pandas.DateTime
        The start date of prediction
    end_future_date: Pandas.DateTime
        The end date of prediction
    all_models: List of TimeseriesForecastingModel
        All model classes using for forecasting
    n_cv: int
        Number folds of cross validation
    length_cv: int
        The length of cv part
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.

    Returns
    -------
    List of dict:
        The list of prediction value, the interval of errors and other related info
    
    """
    future_preds = []
    flat_result_cv, mape_cv, wape_cv, std_cv, model \
        = select_best_model(data_series, all_models, n_cv, length_cv, flag=flag)

    if model is None:
        return None

    preds = model.forecast(start_future_date, end_future_date)
    preds = preds.round()
    
    if flag == 'agg':
        preds = pd.Series(preds.sum(), index = [preds.index[-1]])

    predict_time = list(preds.index)
    predict_value = preds.values
    for time, value in zip(predict_time, predict_value):
        result = {'predict_time': time.isoformat(), 'predict_value': value, 'best_model': model.getLabel(), 
                  'mape_cv': mape_cv, 'wape_cv': wape_cv, 'below_error': mape_cv - std_cv, 'upper_error': mape_cv + std_cv}
        future_preds.append(result)
    return future_preds


def all_forecast_process(total_by_date, freq_, last_day, all_models, length_test, n_cv, length_cv,
                         start_future_date, end_future_date, flag=None, mode='test'):
    """
    All forecast process for a time series
    
    Parameters
    ----------
    total_by_date: Spark.DataFrame
        Dataframe includes information about sale, with 2 columns
    freq_: String
        The aggregation type, can be D, W-SUN, M
    last_day:
        The maximum date of transaction
    all_models: List of TimeseriesForecastingModel
        All model classes using for forecasting
    n_cv: int
        Number folds of cross validation
    length_cv: int
        The length of cv part
    start_future_date: Pandas.DateTime
        The start date of prediction
    end_future_date: Pandas.DateTime
        The end date of prediction
    flag : str, Options: [None, 'agg']
        If flag is agg, actual and predicted values will be aggregated before measuring error.
    mode : str, Options: ['test', 'use']
        Mode of function.
        
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part
    Dict:
        All information of forecasting result for test part with using the best model which get from CV process
    List of dict:
        The list of history value
    List of dict:
        The list of prediction value, the interval of errors and other related info
    """
    if total_by_date.shape[0] == 0:
        return None, None, None, None
    df_clean = clean_data(total_by_date, last_day)
    df_clean = df_clean['daily_quantity'].reset_index()
    if df_clean.shape[0] == 0:
        return None, None, None, None
    df_group = df_clean.groupby(pd.Grouper(key ='transaction_date', freq = freq_))['daily_quantity'].sum()
    df_group = pd.DataFrame(df_group)
    df_group['moy'] = df_group.index.to_series().dt.month
    df_group = pd.get_dummies(df_group, prefix = ['moy'], columns = ['moy'], drop_first = True)
    if (freq_ == 'W-SUN') | (freq_ == 'D'):
        df_group['dom'] = df_group.index.to_series().dt.day
        df_group = pd.get_dummies(df_group, prefix = ['dom'], columns = ['dom'], drop_first = True)
    if freq_ == 'D':
        df_group['dow'] = df_group.index.to_series().dt.dayofweek
        df_group = pd.get_dummies(df_group, prefix = ['dow'], columns = ['dow'], drop_first = True)
    data_series = df_group
    
    if mode == 'test':
        flat_result_cv, flat_result_test = perform_test_process(data_series, all_models, length_test, n_cv, length_cv, flag = flag)
        future_preds = retrain_and_predict(data_series, start_future_date, end_future_date, all_models, n_cv, length_cv, flag = flag)
        if future_preds != None:
            mape_test = None
            if flat_result_test:
                mape_test = flat_result_test['mape_test']
            for pred in future_preds:
                pred.update({'last_mape_error': mape_test})
    elif mode == 'use':
        future_preds = retrain_and_predict(data_series, start_future_date, end_future_date, all_models, n_cv, length_cv, flag = flag)
    else:
        raise Exception('Mode of function is test or use')

    history_data = []
    hist_time = list(df_group.index)
    hist_value = df_group['daily_quantity']
    for time, value in zip(hist_time, hist_value):
        result = {'history_time': time.isoformat(), 'history_value': value}
        history_data.append(result)
    if mode == 'use':
        return history_data, future_preds
    
    return flat_result_cv, flat_result_test, history_data, future_preds


def adaptive_forecast_process(total_by_date, freq_):
    """
    Adptive forecast process when are given a time series and frequency type
    
    Parameters
    ----------
    total_by_date: Spark.DataFrame
        Dataframe includes information about sale, with 2 columns
    freq_: String
        The aggregation type, can be D, W-SUN, M
    
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part
    Dict:
        All information of forecasting result for test part with using the best model which get from CV process
    List of dict:
        The list of history value
    List of dict:
        The list of prediction value, the interval of errors and other related info
    """
    total_by_date.columns = ['transaction_date', 'daily_quantity']
    last_day = total_by_date['transaction_date'].max()
    # current_day = pd.to_datetime('today') - pd.DateOffset(days=1)
    # current_day = current_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    current_day = last_day
    length_test = 15
    n_cv = 3
    start_future_date = current_day + pd.DateOffset(days=1)
    end_future_date = current_day + pd.DateOffset(days=15)
    if freq_ == 'W-SUN':
        current_day = current_day - pd.DateOffset(days=current_day.dayofweek + 1)
        length_test = 1
        start_future_date = current_day + pd.DateOffset(days=7)
        end_future_date = current_day + pd.DateOffset(days=28)
    elif freq_ == 'M':
        if current_day.day < 28:
            current_day = current_day - pd.DateOffset(days=current_day.day)
        length_test = 1
        current_month = current_day.month
        end_day_months = pd.date_range(f'{current_day.year}-01-01', f'{current_day.year}-12-01', freq = 'M')
        # get last date of next month
        start_future_date = end_day_months[current_month-1]  + pd.DateOffset(months=2)
        start_future_date = start_future_date - pd.DateOffset(days=start_future_date.day)
        # get last date of next 3rd month
        end_future_date = start_future_date + pd.DateOffset(months=3)
        end_future_date = end_future_date - pd.DateOffset(days=end_future_date.day)

    length_cv = length_test
    all_models = {'Average': AverageModel, 'Naive': NaiveModel, 'Drift': DriftModel,
                  'SeasonalNaive':SeasonalNaiveModel, 'UpgradeSeason':UpgradeSeasonModel,
                  'StandardArima': StandardArimaModel, 'SES':SimpleExpSmothingModel,
                  'Holt': HoltModel, 'HoltWinter': HoltWinterModel,
                  'DecomposeArima': DecomposeArimaModel, 'Prophet': ProphetModel}
    return all_forecast_process(total_by_date, freq_, current_day, all_models, length_test, n_cv, length_cv,
                                start_future_date, end_future_date)


def forecasting_aggregation_process(total_by_date, start_forecast_day = 'today', n_days = 15, mode = 'test'):
    """
    Forecast daily then aggregate by number of days.
    
    Parameters
    ----------
    total_by_date: Spark.DataFrame
        Dataframe includes information about sale, with 2 columns: ['transaction_date', 'daily_quantity']
    start_forecast_day : str
        Start day of forecast. Default: "today". Different day should follow format yyyy-mm-dd.
    n_days: int
        Number of days that is made prediction for, including start_forecast_day.
    mode : str, Options: ['test', 'use']
        Mode of function.
    Returns
    -------
    List of dict:
        All information of forecasting result for each model and each fold in CV part. If mode is test.
    Dict:
        All information of forecasting result for test part with using the best model which get from CV process. If mode is test.
    List of dict:
        The list of history value
    List of dict:
        The list of prediction value, the interval of errors and other related info
    """
    total_by_date.columns = ['transaction_date', 'daily_quantity']
    max_day = total_by_date['transaction_date'].max()
    current_day = pd.to_datetime(start_forecast_day) - pd.DateOffset(days=1)
    current_day = current_day.replace(hour = 0, minute = 0, second = 0, microsecond = 0)
    if current_day > max_day:
        raise Exception(current_day, 'is not in the dataset.')
    length_test = n_days
    n_cv = 3
    start_future_date = current_day + pd.DateOffset(days=1)
    end_future_date = current_day + pd.DateOffset(days=n_days)
    length_cv = length_test
    all_models = {'Average': AverageModel, 'Naive': NaiveModel, 'SeasonalNaive':SeasonalNaiveModel,
                  # 'UpgradeSeason':UpgradeSeasonModel, 'StandardArima': StandardArimaModel,
                  # 'SES':SimpleExpSmothingModel, 'Drift': DriftModel, 'DecomposeArima': DecomposeArimaModel,
                  'Holt': HoltModel, 'HoltWinter': HoltWinterModel, 'Prophet': ProphetModel}
    if mode == 'use':
        return all_forecast_process(total_by_date = total_by_date, freq_ = 'D',
                                    last_day= current_day, all_models = all_models,
                                    length_test = length_test, n_cv = n_cv, length_cv = length_cv,
                                    start_future_date = start_future_date, end_future_date = end_future_date,
                                    flag = 'agg', mode = mode)
    return all_forecast_process(total_by_date = total_by_date, freq_ = 'D',
                                last_day= current_day, all_models = all_models,
                                length_test = length_test, n_cv = n_cv, length_cv = length_cv,
                                start_future_date = start_future_date, end_future_date = end_future_date,
                                flag = 'agg')
