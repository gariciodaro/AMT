import pandas as pd
# Call imputer, to fill misssing values
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class DataStandardizer:


    def __init__(self):
        pass

    @staticmethod
    def fix_columns_and_decimal(df):
        cols=list(df.columns)
        mapper_cols={str(index):each_col for index,each_col in enumerate(cols)}
        df_ajusted=pd.DataFrame(df.values, columns=list(mapper_cols.keys()),dtype=df.values.dtype)
        df_temp=df_ajusted.copy()
        for every_col in list(df_ajusted.columns):
            try:
                df_temp[every_col] = [x.replace(',', '.') for x in df_ajusted[every_col]]
                df_temp[every_col] = df_temp[every_col].astype(float)
            except:
                try:
                    df_temp[every_col] = df_temp[every_col].astype(float)
                except:
                    pass
        return df_temp,mapper_cols

    @staticmethod
    def get_numerics_cols(df):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        df_numeric = df.select_dtypes(include=numerics)
        return df_numeric

    @staticmethod
    def replace_nan(df):
        imputer = Imputer(missing_values= 'NaN', strategy = 'mean',axis=0)
        imputer_values=imputer.fit_transform(df)
        df_nonan=pd.DataFrame(imputer_values,columns=df.columns,index=df.index)
        return df_nonan


    @staticmethod
    def get_binary_target(df_ajusted,target_col):
        df=df_ajusted[[target_col]].copy()
        labels=list(df[target_col].unique())
        labels.sort()
        df.replace(labels[0], 0,inplace=True)
        df.replace(labels[1], 1,inplace=True)
        mapped_label={labels[0]:0,labels[1]:1}
        return df,mapped_label

    @staticmethod
    def get_null_coll(df):
        null_cols_list=df.isna().any()
        return null_cols_list


    @staticmethod
    def get_encoded_single_df(data_frame_column):
        """one hot encode a single column dataframe.
        Args:
            data_frame_column: single column datafrar.

        Returns:
            out: list. [0]:encoded array. [1] fitted object encoder
        """
        df_initial=data_frame_column.copy()
        encoder_object=OneHotEncoder(sparse=False,categories='auto',drop='first')
        data_frame_column=np.array(data_frame_column)
        data_frame_column=data_frame_column.reshape(-1, 1)
        #print(data_frame_column.shape)
        encoded_data = encoder_object.fit_transform(data_frame_column)
        cat=encoder_object.categories_
        #print("----",cat[0])
        cat=cat[0][1:]
        #print("----",cat)
        df_encoded_data=pd.DataFrame(encoded_data,columns=cat,index=df_initial.index)
        return[df_encoded_data,encoder_object]

    @staticmethod
    def transform_encoded_single_df(data_frame_column,encoder_object):
        df_initial=data_frame_column.copy()

        data_frame_column=np.array(data_frame_column)
        data_frame_column=data_frame_column.reshape(-1, 1)
        #print(data_frame_column.shape)
        encoded_data = encoder_object.transform(data_frame_column)
        df_encoded_data=pd.DataFrame(encoded_data,columns=encoder_object.categories_,index=df_initial.index)
        return df_encoded_data
    
    def pipe_line_stand(self,df,mode="target"):

        df,maped_cols_1=self.fix_columns_and_decimal(df)
        if mode=="target":
            target_1=str(len(maped_cols_1)-1)
            Y_1,maped_labels_1=self.get_binary_target(df,target_1)
        original_cols=list(df.columns)
        df_hols=df.copy()
        df=self.get_numerics_cols(df)
        df=self.replace_nan(df)
        if mode=="target":
            df[target_1]=Y_1
            return df,target_1
        else:
            return df





