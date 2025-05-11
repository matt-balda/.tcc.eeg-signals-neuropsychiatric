import pandas as pd
from sklearn.impute import KNNImputer

class DataCleaner:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        #self.__remove_missing_columns()
    
    def remove_missing_columns(self, threshold=0.5):
        limit = int(threshold * len(self.df))
        self.df = self.df.dropna(thresh=limit, axis=1)
        
    def find_most_null_column(self, threshold=0.5):
        null_ratios = self.df.isnull().mean()
        for col, ration in null_ratios.items():
            if ration > threshold:
                return col
        return None

    def analyze_missing_values(self):
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        total_number_nans = self.df.isnull().sum().sum()
        
        return missing_values, total_number_nans
    
    def handle_nans(self):
        columns_with_nans = self.df.columns[self.df.isnull().any()].tolist()
        
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
        
        df_imputed = pd.DataFrame(knn_imputer.fit_transform(self.df[columns_with_nans]),
                                  columns=columns_with_nans)
        
        self.df[columns_with_nans] = df_imputed[columns_with_nans]
        
    def get_dataframe(self):
        return self.df
