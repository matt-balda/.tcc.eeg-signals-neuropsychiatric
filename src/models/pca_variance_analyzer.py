import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PCAVarianceAnalyzer:
    def __init__(self, variance_thresholds=(0.95, 0.99)):
        self.variance_thresholds = variance_thresholds
        self.scaler = StandardScaler()
        self.pca = None
        self.explained_variance_ = None
        self.cumulative_variance_ = None
        self.n_components_ = {}
    
    def fit(self, X):

        X_scaled = self.scaler.fit_transform(X)
        
        self.pca = PCA()
        self.pca.fit(X_scaled)
        
        self.explained_variance_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = self.explained_variance_.cumsum()
        
        for threshold in self.variance_thresholds:
            n_components = (self.cumulative_variance_ >= threshold).argmax() + 1
            self.n_components_[threshold] = n_components
            print(f"Components required for {int(threshold*100)}% of variance: {n_components}")

        self._plot_cumulative_variance()

        return X_scaled

    def transform(self, X, threshold=0.99):
        if threshold not in self.n_components_:
            raise ValueError(f"Threshold {threshold} not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        pca = PCA(n_components=self.n_components_[threshold])
        X_pca = pca.fit_transform(X_scaled)

        print(f"Shape of the data after PCA with {self.n_components_[threshold]} components: {X_pca.shape}, type: {type(X_pca)}")
        
        return X_pca
    
    def _plot_cumulative_variance(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.cumulative_variance_, marker='o', linestyle='--')
        plt.title('Cumulative Explained Variance - PCA')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.show()
