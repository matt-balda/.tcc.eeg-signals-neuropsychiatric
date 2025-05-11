import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE

class DataPreparer:
    def __init__(self, dataframe: pd.DataFrame, target_column: str):
        self.df = dataframe.copy()
        self.target_column = target_column
    
    def minority_class_balancer(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        borderline_smote = BorderlineSMOTE(k_neighbors=3, random_state=42)
        X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
        
        self.df = pd.DataFrame(X_resampled, columns=X.columns)
        self.df[self.target_column] = y_resampled
        
    def get_dataframe(self):
        return self.df
