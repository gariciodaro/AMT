from auxiliar.classes.StaticML import StaticML
from auxiliar.classes.DataStandardizer import DataStandardizer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class Predictor():

    def __init__(self,best_clf,min_max_scaler):
        self.best_clf=best_clf
        self.min_max_scaler=min_max_scaler

    def prepare_for_predict(self,X):
        X = StaticML.transform_min_max(X,self.min_max_scaler)
        return X


    def make_predictions(self,df):
        Ds=DataStandardizer()
        #Remember to pass only features, not the target!
        df_vals=Ds.pipe_line_stand(df,mode="no_target")

        self.X=self.prepare_for_predict(df_vals)

        prediction=self.best_clf.predict_proba(self.X)[:,1]

        df_prediction=pd.DataFrame(prediction,index=self.X.index,columns=["Predicted_Target"])

        return df_prediction