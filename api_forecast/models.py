import matplotlib.pyplot as plt
from scipy.signal import periodogram
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as scs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
import math
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults, ARMAResults, ARIMAResultsWrapper
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from fbprophet import Prophet

# from keras import Sequential
# from keras.layers import LSTM, Dense, Dropout

class TimeseriesForecastingModel(ABC):
    """
    Abstract class for all time series forecasting model

    Pararmeters
    -----------
    label: string
        The name of model
    train_series: Pandas.DataFrame
        The beforehand given time series
    predictor_cols: array-like
        All predictor columns of train set
    target_col: array_like
        The response column of train set
    model: Tuple
        The achieved model after fitting
        
    Returns
    -------
    timeseries_forecasting_model class
    
    """
    def __init__(self, label, train_series, predictor_cols, target_col):
        self.train_series = train_series
        self.predictor_cols = predictor_cols
        self.target_col = target_col
        self.label = label
        self.model = None
        self.resid = None
        if self.train_series.index.freq == 'D':
            self.freq = 1
        elif self.train_series.index.freq == 'M':
            self.freq = 28
        else:
            self.freq = 7
            
    
    def getLabel(self):
        return self.label
    
    def _gen_random_sample(self, n, length):
        """
        Generate n random start points 
        
        """
        rand_start_points = np.random.randint(1, self.train_series.shape[0] - length, n)
        return rand_start_points
    
    @abstractmethod
    def fit(self):
        """
        Fit the model
        
        Returns
        -------
        The parameters of model
        """
    @abstractmethod
    def forecast(self, start_date, end_date):
        """
        Forecast for test series
        
        Parameters
        ----------
        start_date: DateType
            Start day for prediction
        end_date: DateType
            End day for prediction
        exog: 
            Additional argument may required
        
        Returns
        -------
        Pandas.Series:
            Prediction for the test series
        """

    def measure_error(self, test_series, preds):
        """
        Get the error information when giving prediction and actual value
        
        Parameters
        ----------
        test_series: Pandas.DataFrame
            The timeseries of test set
        preds: Pandas.Series
            The prediction values for test set
            
        Returns
        -------
        Pandas.DataFrame:
            Information of error value for each period
        Dict:
            Total error value
        """
        target_col = self.target_col
        result_columns = ['period', 'demand', 'forecast', 'error', 'abs_error', 'percent_error']
        result = pd.DataFrame([], columns = result_columns)
        n = 0
        total_mse = 0
        total_mad = 0
        total_mape = 0
        for index, row in test_series.iterrows():
            pred = preds.values[n]
            error = pred - row[target_col].values[0]
            abs_error = math.fabs(error)
            if row[target_col].values[0] == 0:
                percent_error = abs_error
            else:
                percent_error = abs_error/math.fabs(row[target_col].values[0])
            n = n + 1
            total_mse += error ** 2
            total_mad += abs_error
            total_mape += percent_error
            v = [index, row[target_col].values[0], pred, error, abs_error, percent_error]
            result = result.append([dict(zip(result_columns, v))])
        result.index = np.arange(test_series.shape[0])
        mse = total_mse/n
        mad = total_mad/n
        mape = total_mape/n
        total_error = {'MSE': mse, 'MAD': mad, 'MAPE': mape}
        return result, total_error
    
    def cross_validation(self, n_cv, length_test):
        """
        Evaluate model on the train set by cross validation
        
        Parameters
        ---------
        n_cv: int
            Number folds for validation
        length_test: int
            The length of one fold
        
        Returns
        -------
        List:
            Time series prediction for all folds
        Pandas.DataFrame:
            Average error across all folds
        Dict:
            Average total error across all folds
            
        """
        rand_start_points = self._gen_random_sample(n_cv, length_test)
        result_columns = ['demand', 'forecast', 'error', 'abs_error', 'percent_error']
        total_cv_error = pd.DataFrame(np.zeros((length_test, len(result_columns))), columns = result_columns)
        list_cv_preds = []
        total_agg_error = {'MSE': 0, 'MAD': 0, 'MAPE': 0}
        for start_point in rand_start_points:
            cv_series = self.train_series.iloc[start_point : start_point+length_test]
            start_date = cv_series.index[0]
            end_date = cv_series.index[-1]
            cv_preds_series= self.forecast(start_date, end_date)
            cv_error_df, cv_agg_error = self.measure_error(cv_series, cv_preds_series)
            cv_error_df = cv_error_df.reset_index()[result_columns]
            for key in total_agg_error:
                total_agg_error[key] += cv_agg_error[key]
            list_cv_preds.append(cv_preds_series)
            total_cv_error = total_cv_error + cv_error_df
        avg_cv_error = total_cv_error/n_cv
        for key in total_agg_error:
            total_agg_error[key] = total_agg_error[key]/n_cv
        return list_cv_preds, avg_cv_error, total_agg_error
    
    def _get_exog(self, start_date, end_date):
        exog = pd.DataFrame(index = pd.date_range(start_date, end_date, freq = self.train_series.index.freq))
        exog['dow'] = exog.index.to_series().dt.dayofweek
        exog['dom'] = exog.index.to_series().dt.day
        exog['month'] = exog.index.to_series().dt.month
        exog = pd.get_dummies(exog, prefix = ['dow', 'moy', 'dom'], columns = ['dow', 'month', 'dom'])
        left_columns = list(set(self.predictor_cols) - set(exog.columns))
        for col in left_columns:
            exog[col] = np.zeros(exog.shape[0])
        return exog[self.predictor_cols]
    
class AverageModel(TimeseriesForecastingModel):
    """
    Using average value for any prediction 
    """
    
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Average', train_series, predictor_cols, target_col)

    def fit(self):
        average = np.mean(self.train_series[self.target_col])
        self.model = (average,)
        self.resid = average - self.train_series[self.target_col]
        
    def forecast(self, start_day, end_day):
        if not self.model:
            self.fit()
        index_ = pd.date_range(start_day, end_day, freq = self.train_series.index.freq)
        number_days = len(index_)
        preds = np.full(number_days, self.model[0])
        preds_series = pd.Series(preds, index = index_)
        return preds_series
        
class NaiveModel(TimeseriesForecastingModel):
    """
    Using the last value of train for prediction
    """
    
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Naive', train_series, predictor_cols, target_col)

    def fit(self):
        last_value = self.train_series.iloc[-1][self.target_col[0]]
        self.model = (last_value,)
        self.resid = last_value - self.train_series[self.target_col[0]]
        
    def forecast(self, start_day, end_day):
        if not self.model:
            self.fit()
        last_day = start_day - pd.Timedelta(f'{self.freq} days')
        if last_day not in self.train_series.index:
            last_day = self.train_series.index[-1]
        last_value = self.train_series.loc[last_day][self.target_col[0]]
        index_ = pd.date_range(start_day, end_day, freq = self.train_series.index.freq)
        number_days = len(index_)
        preds = np.full(number_days, last_value)
        preds_series = pd.Series(preds, index = index_)
        return preds_series

class DriftModel(TimeseriesForecastingModel):
    """
    Using time index as the predictor for forecastning
    """
    
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Drift', train_series, predictor_cols, target_col)
        
    def fit(self):
        X = np.arange(self.train_series.shape[0]).reshape(-1,1)
        y = np.array(self.train_series[self.target_col]).ravel()
        self.model = LinearRegression().fit(X, y)
        self.resid = pd.Series(self.model.predict(X).ravel() - y, self.train_series.index)
    
    def forecast(self, start_day, end_day):
        if not self.model:
            self.fit()
        last_day = start_day - pd.Timedelta(f'{self.freq} days')
        if last_day not in self.train_series.index:
            start_index = self.train_series.shape[0] + (last_day - self.train_series.index[-1]).days//self.freq
        else:
            start_index = self.train_series.index.get_loc(last_day) + 1
        index_ = pd.date_range(start_day, end_day, freq = self.train_series.index.freq)
        number_days = len(index_)
        x_ = np.arange(start_index, start_index + number_days).reshape(-1, 1)
        preds = self.model.predict(x_).ravel()
        preds_series = pd.Series(preds, index = index_)
        return preds_series
    
def find_summary_arima(train_series, p, d, q):
    try:
        model = ARIMA(train_series, (p, d, q), freq = train_series.index.freq)
        result_model = model.fit()
        aic_value = result_model.aic
        return aic_value, result_model
    except:
        return None, None
    
def find_best_arima_order(train_series):
    rng = range(3)
    best_aic = np.inf
    best_order = None
    best_result_mdl = None
    for p in rng:
        for d in range(2):
            for q in rng:
                aic_value, result_mdl = find_summary_arima(train_series, p, d, q)
                if (aic_value != None):
                    if (aic_value < best_aic):
                        best_aic = aic_value
                        best_order = (p, d, q)
                        best_result_mdl = result_mdl
    return best_order, best_result_mdl

class SeasonalNaiveModel(TimeseriesForecastingModel):
    """
    Using time index and predictor columns in regression
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'SeasonalNaive', train_series, predictor_cols, target_col)
    def fit(self):
        X = self.train_series[self.predictor_cols]
        X['index'] = np.arange(self.train_series.shape[0])
        X = np.array(X)
        y = np.array(self.train_series[self.target_col]).ravel()
        self.model = LinearRegression().fit(X, y)
        self.resid = pd.Series(self.model.predict(X).ravel() - y, self.train_series.index)
        
    def forecast(self, start_date, end_date):
        x_ = self._get_exog(start_date, end_date)
        if (not self.predictor_cols):
            x_ = pd.DataFrame()
        if not self.model:
            self.fit()
        index_ = pd.date_range(start_date, end_date, freq = self.train_series.index.freq)
        number_days = len(index_)
        last_day = start_date - pd.Timedelta(f'{self.freq} days')
        if last_day not in self.train_series.index:
            start_index = self.train_series.shape[0] + (last_day - self.train_series.index[-1]).days//self.freq
        else:
            start_index = self.train_series.index.get_loc(last_day) + 1
        x_['index'] = np.arange(start_index, start_index + number_days)
        x_ = np.array(x_)
        preds = self.model.predict(x_).ravel()
        preds_series = pd.Series(preds, index = index_)
        return preds_series
    
class UpgradeSeasonModel(TimeseriesForecastingModel):
    """
    Using time index and predictor columns in regression. ARIMA for predict the one after remove seasonal pattern
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'UpgradeSeason', train_series, predictor_cols, target_col)
    
    def fit(self):
        X = self.train_series[self.predictor_cols]
        X['index'] = np.arange(self.train_series.shape[0])
        X = np.array(X)
        y = np.array(self.train_series[self.target_col]).ravel()
        linear_model = LinearRegression().fit(X, y)
        linear_resid = pd.Series(linear_model.predict(X).ravel() - y, self.train_series.index)
        best_order, arima_model = find_best_arima_order(linear_resid)
        self.resid = arima_model.resid
        self.model = (linear_model, arima_model)
        
    def forecast(self, start_date, end_date):
        x_ = self._get_exog(start_date, end_date)
        if (not self.predictor_cols):
            x_ = pd.DataFrame()
        if not self.model:
            self.fit()
        number_days = len(pd.date_range(start_date, end_date, freq = self.train_series.index.freq))
        last_day = start_date - pd.Timedelta(f'{self.freq} days')
        if last_day not in self.train_series.index:
            start_index = self.train_series.shape[0] + (last_day - self.train_series.index[-1]).days//self.freq
        else:
            start_index = self.train_series.index.get_loc(last_day) + 1
        x_['index'] = np.arange(start_index, start_index + number_days)
        x_ = np.array(x_)
        seasonal_preds = self.model[0].predict(x_).ravel()
        if (type(self.model[1]) == ARIMAResults):
            trend_residual_preds = self.model[1].predict(start_date, end_date, typ = 'levels')
        else:
            trend_residual_preds = self.model[1].predict(start_date, end_date)
        preds_series  = trend_residual_preds + seasonal_preds
        return preds_series
    
class StandardArimaModel(TimeseriesForecastingModel):
    """
    Using standard Arima
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'StandardArima', train_series, predictor_cols, target_col)
    
    def fit(self):
        best_order, arima_model = find_best_arima_order(self.train_series[self.target_col])
        self.resid = arima_model.resid
        self.model = (arima_model,)
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        if ((type(self.model[0]) == ARIMAResultsWrapper) | (type(self.model[0]) == ARIMAResults)):
            preds_series = self.model[0].predict(start_date, end_date, typ = 'levels')
        else:
            preds_series = self.model[0].predict(start_date, end_date)
        return preds_series
    
class SimpleExpSmothingModel(TimeseriesForecastingModel):
    """
    Using simple exponential smoothing model
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'SES', train_series, predictor_cols, target_col)
    
    def fit(self):
        model = SimpleExpSmoothing(self.train_series[self.target_col]).fit()
        self.model = (model,)
        self.resid = model.resid
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        preds_series = self.model[0].predict(start_date, end_date)
        return preds_series
    
class HoltModel(TimeseriesForecastingModel):
    """
    Using Holt model
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Holt', train_series, predictor_cols, target_col)
    
    def fit(self):
        model = Holt(self.train_series[self.target_col]).fit()
        self.model = (model,)
        self.resid = model.resid
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        preds_series = self.model[0].predict(start_date, end_date)
        return preds_series
    
class HoltWinterModel(TimeseriesForecastingModel):
    """
    Using Holt Winter model
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'HoltWinter', train_series, predictor_cols, target_col)
    
    def fit(self):
        if self.train_series.index.freq == 'M':
            season_periods_  = 12
        else:
            season_periods_ = 7
        model = ExponentialSmoothing(self.train_series[self.target_col], trend = 'add', seasonal = 'add', seasonal_periods = season_periods_).fit()
        self.model = (model,)
        self.resid = model.resid
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        preds_series = self.model[0].predict(start_date, end_date)
        return preds_series
    
class DecomposeArimaModel(TimeseriesForecastingModel):
    """
    Using decompose to extract the seasonal pattern from the time series, then using ARIMA for predict the rest
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'DecomposeArima', train_series, predictor_cols, target_col)
        self.model = None
        self.resid = None
        
    def fit(self):
        n_train = self.train_series.shape[0]
        freq_ = 7
        if (self.train_series.index.freq == 'M'):
            freq_ = 12
        decompose_parts = sm.tsa.seasonal_decompose(self.train_series[self.target_col], freq = freq_, model = 'additive')
        remove_seasonal = decompose_parts.observed - decompose_parts.seasonal
        best_order, arima_model = find_best_arima_order(remove_seasonal)
        self.model = (decompose_parts.seasonal.values[:,0], arima_model)
        self.resid = arima_model.resid
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        number_days = len(pd.date_range(start_date, end_date, freq = self.train_series.index.freq))
        if (self.train_series.index.freq == 'M'):
            last_day = start_date - pd.Timedelta(f'{start_date.day} days')
            freq_ = 12
        else:
            freq_ = 7
            last_day = start_date - pd.Timedelta(f'{self.freq} days')
        if last_day not in self.train_series.index:
            start_index = self.train_series.shape[0] + (last_day - self.train_series.index[-1]).days//self.freq
        else:
            start_index = self.train_series.index.get_loc(last_day) + 1
        seasonal_preds = [self.model[0][(start_index + i) % freq_] for i in range(number_days)]
        if ((type(self.model[1]) == ARIMAResultsWrapper) | (type(self.model[1]) == ARIMAResults)):
            trend_residual_preds = self.model[1].predict(start_date, end_date, typ = 'levels')
        else:
            trend_residual_preds = self.model[1].predict(start_date, end_date)
        preds_series = trend_residual_preds + seasonal_preds
        return preds_series
    
class SarimaxModel(TimeseriesForecastingModel):
    """
    Using Sarimax
    """
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Sarimax', train_series, predictor_cols, target_col)
        
    def fit(self):
        if self.predictor_cols:
            exog_train = self.train_series[self.predictor_cols].values
        else:
            exog_train = None
        X = self.train_series[self.target_col]
        order_ = (2, 1, 2, 7)
        if (self.train_series.index.freq == 'M'):
            order_ = (1, 1, 1, 12)
        model = SARIMAX(X, exog=exog_train, order=(2,1,2), seasonal_order=order_, \
                        time_varying_regression=True, mle_regression=False, enforce_stationarity = False, enforce_invertibility=False).fit()
        self.model = (model,)
        self.resid = model.resid
        
    def forecast(self, start_date, end_date):
        if not self.model:
            self.fit()
        if self.predictor_cols:
            exog_test= self._get_exog(start_date, end_date).values
        else:
            exog_test= None
        preds_series = self.model[0].predict(start_date, end_date, exog = exog_test)
        return preds_series
    
class ProphetModel(TimeseriesForecastingModel):
    """
    Using Prophet model for prediction
    """
    
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'Prophet', train_series, predictor_cols, target_col)
    
    def fit(self):
        """
        Fitting Prohphet model
        """
    
        df_train = self.train_series[self.target_col].reset_index()
        df_train.columns = ['ds', 'y']
        self.model = Prophet()
        self.model.fit(df_train)
        
    def forecast(self, start_date, end_date):
        future = pd.DataFrame(pd.date_range(start_date, end_date, freq = self.train_series.index.freq))
        future.columns = ['ds']
        preds = self.model.predict(future)[['ds', 'yhat']].set_index('ds')
        preds_series = preds['yhat']
        return preds_series
    
class LstmModel(TimeseriesForecastingModel):
    """
    Using LSTM model for prediction 
    """
    
    def __init__(self, train_series, predictor_cols, target_col):
        TimeseriesForecastingModel.__init__(self, 'LSTM', train_series, predictor_cols, target_col)

    def fit(self):
        """
        Fitting LSTM model
        """
        
        # Feature scaling training set
        self.scaler = MinMaxScaler(feature_range = (0, 1))
        training_set = self.train_series[self.target_col].values.reshape(-1, 1)
        training_set_scaled = self.scaler.fit_transform(training_set)
        
        # Creating a data structure with number of timesteps
        self.n_timesteps = 7
        n_training_samples = training_set_scaled.shape[0]

        X_train = []
        y_train = []

        for i in range (self.n_timesteps, n_training_samples):
            X_train.append(training_set_scaled[i - self.n_timesteps: i])
            y_train.append(training_set_scaled[i])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Initialising LSTM model architecture
        self.model = Sequential()
        
        self.model.add(LSTM(units = 32, activation = 'relu', input_shape = (X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(units = 1))
        
        # Training the model
        self.model.compile(optimizer = 'adam', loss = 'mse')
        self.model.fit(X_train, y_train, epochs = 300, batch_size = 32, verbose = 0)
        
        y_train_true = self.scaler.inverse_transform(y_train)
        y_train_pred = self.scaler.inverse_transform(self.model.predict(X_train))
        self.resid = y_train_pred - y_train_true
        
    def forecast(self, start_day, end_day):
        """
        Forecasting using fitted model
        """
        if not self.model:
            self.fit()
        number_days = (end_day - start_day).days + 1
        
        train_start_day = self.train_series.index[0]
        train_end_day = self.train_series.index[-1]
        
        if start_day <= train_end_day:
            # Input for forecasting on CV
            if (start_day - train_start_day).days < self.n_timesteps:
                raise Exception("Validation set should have equal or more than {} previous days".format(self.n_timesteps))
            inputs = self.train_series[start_day - timedelta(self.n_timesteps): end_day][self.target_col].values.reshape(-1, 1)
            inputs = self.scaler.transform(inputs)
            
            # Validation set of X and y
            X_val = []
            y_val = []
            n_input_samples = inputs.shape[0]

            for i in range(self.n_timesteps, n_input_samples):
                X_val.append(inputs[i - self.n_timesteps: i, 0])
                y_val.append(inputs[i, 0])

            X_val, y_val = np.array(X_val), np.array(y_val)
            X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
            
            # Prediction on CV
            preds = self.model.predict(X_val)
        else:
            # Input for forecasting on Test set
            inputs = self.train_series[-self.n_timesteps:][self.target_col].values.reshape(-1, 1)
            inputs = self.scaler.transform(inputs)
            n_test = number_days

            # Forecasting for number of days
            for i in range(n_test):
                X_test = [inputs[-self.n_timesteps:]]
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                y_pred_i = self.model.predict(X_test)
                inputs = np.append(inputs, y_pred_i, axis = 0)
            preds = inputs[-n_test:]
        
        # Transform predictions to series
        preds = self.scaler.inverse_transform(preds)
        preds = np.reshape(preds, (preds.shape[0],))
        preds_series = pd.Series(preds, index = pd.date_range(start_day, end_day))
        return preds_series