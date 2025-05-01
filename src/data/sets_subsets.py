import pandas as pd

class Sets:
    # AB = PSD (Power Spectral Density) 19 * 6
    # COH = FC (Functional Connectivity) 171 * 6
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 quantitative_features: list[str],
                 qualitative_features: list[str],
                 target_main: str,
                 target_specific: str):
        
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("O parâmetro precisa ser um pandas DataFrame.")
        
        self.df = dataframe.copy()
        self.quantitative_features = self.df[quantitative_features]
        self.qualitative_features = self.df[qualitative_features]
        self.target_main = self.df[[target_main]]
        self.target_specific = self.df[[target_specific]]
        self.df_ab_psd = None
        self.df_coh_fc = None
        self.df_ab_psd_coh_fc = None
        #self.dfs_bands = {}
        
        self.__create_df_psd()
        self.__create_df_fc()
        self.__create_union_psd_fc()

    # 19 (6 bands)
    def __create_df_psd(self):
        columns_AB = [col for col in self.df.columns if col.startswith('AB')]
        self.df_ab_psd = self.df[columns_AB]
    
    # 171 (6 bands)
    def __create_df_fc(self):
        columns_COH = [col for col in self.df.columns if col.startswith('COH')]
        self.df_coh_fc = self.df[columns_COH]
        
    def __create_union_psd_fc(self):
        if self.df_ab_psd is not None and self.df_coh_fc is not None:
            self.df_ab_psd_coh_fc = pd.concat([self.df_ab_psd, self.df_coh_fc], axis=1)
        else:
            raise ValueError("Os subconjunto AB e COH não foram criados corretamente.")

    def create_dfs_bands(self, bands: list[str] = None, df: pd.DataFrame = None):
        dfs_bands = {}
        
        if bands is None:
            bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta','gamma']
        
        for band in bands:
            columns_band = [col for col in df.columns if f'.{band}.' in col]
            if columns_band:
                dfs_bands[band] = df[columns_band]
        
        return dfs_bands