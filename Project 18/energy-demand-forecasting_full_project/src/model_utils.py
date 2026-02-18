import json
import joblib
from pmdarima import auto_arima
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


def auto_sarimax_fit_template(y_series, X_series):
    return auto_arima(
        y=y_series,
        X=X_series,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        seasonal=True, m=6,
        start_P=0, start_Q=0,
        max_P=1, max_Q=1,
        d=None, D=None,
        trace=False,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        information_criterion='aic'
    )

def train_prophet(y_train, X_train):
    df_train = X_train.copy()
    df_train['ds'] = y_train.index
    df_train['y'] = y_train.values

    m = Prophet(seasonality_mode='additive', daily_seasonality=False, yearly_seasonality=True)
    for col in X_train.columns:
        m.add_regressor(col)

    m.fit(df_train)
    return m

def train_lgbm(y_train, X_train):
    m = LGBMRegressor(random_state=42)
    m.fit(X_train, y_train)
    return m

def train_xgboost(y_train, X_train):
    m = XGBRegressor(random_state=42)
    m.fit(X_train, y_train)
    return m

def train_arima(y_train, X_train):
    return auto_arima(
        y=y_train,
        seasonal=False,
        d=None,
        stepwise=True,
        suppress_warnings=True
    )

def train_voting_regressor(models, y_train, X_train):
    estimators = [(name, model) for name, model in models.items() if name not in ['SARIMAX', 'ARIMA', 'PROPHET']]
    m = VotingRegressor(estimators=estimators)
    m.fit(X_train, y_train)
    return m

def train_random_forest(y_train, X_train):
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(y_train, X_train):
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    return model

