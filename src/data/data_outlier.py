import pandas as pd
import numpy as np

class OutlierHandler:
    def __init__(self, df: pd.DataFrame, factor: float = 1.5):
        self.df = df.copy()
        self.factor = factor

    def detect_outliers_summary(self) -> pd.DataFrame:
        outliers_summary = {}

        for col in self.df.select_dtypes(include=['float64', 'int64']).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - self.factor * IQR
            upper_limit = Q3 + self.factor * IQR

            outliers = self.df[(self.df[col] < lower_limit) | (self.df[col] > upper_limit)][col]
            
            outliers_summary[col] = {
                'num_outliers': len(outliers),
                'percent_outliers': len(outliers) / len(self.df) * 100,
                'outliers': outliers.tolist(),
                'lower_limit': lower_limit,
                'upper_limit': upper_limit
            }

        return pd.DataFrame(outliers_summary).T

    def treat_outliers(self) -> pd.DataFrame:
        df_treated = self.df.copy()

        for col in df_treated.select_dtypes(include=[np.number]).columns:
            Q1 = np.percentile(df_treated[col], 25)
            Q3 = np.percentile(df_treated[col], 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR

            df_treated[col] = np.clip(df_treated[col], lower_bound, upper_bound)

        return df_treated
