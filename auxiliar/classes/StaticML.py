
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


class StaticML:
    def __init__(self):
        pass

    @staticmethod
    def min_max_df(df_X):
        scaler = MinMaxScaler()
        X_scaled=scaler.fit_transform(df_X)
        X_scaled=pd.DataFrame(X_scaled,columns=df_X.columns,index=df_X.index)
        return scaler,X_scaled

    @staticmethod
    def transform_min_max(df_X,min_max_scaler):
        X_scaled=min_max_scaler.transform(df_X)
        X_scaled=pd.DataFrame(X_scaled,columns=df_X.columns,index=df_X.index)
        return X_scaled