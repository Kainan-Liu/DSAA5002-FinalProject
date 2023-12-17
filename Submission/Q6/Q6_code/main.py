from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from typing import Optional
import os
import pandas as pd
import matplotlib.pyplot as plt

class Cluster():
    def __init__(self, data_dir) -> None:
        if os.path.exists(data_dir):
            self.data = pd.read_csv(data_dir)
            if self.data.isna().any().any():
                self.data.dropna(inplace=True, axis=0)
        else:
            raise FileNotFoundError
    
    def kmeans_fit(self, features: Optional[list] = None):
        data = self.data.copy()
        if features:
            data = data.loc[:, features]
        # data = pd.get_dummies(data=data)
        kmeans_model =KMeans(n_clusters=8, n_init="auto", random_state=42)
        labels = kmeans_model.fit_predict(data)

        # Visualize the clusters
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title('KMeans Clustering Result')
        plt.show()

    def dbscan_fit(self, features: Optional[list] = None):
        data = self.data.copy()
        if features:
            data = data.loc[:, features]
        # data = pd.get_dummies(data=data)
        dbscan_model = DBSCAN()
        labels = dbscan_model.fit_predict(data)

        # Visualize the clusters
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title('DBSCAN Clustering Result')
        plt.show()

    def hdbscan_fit(self, features: Optional[list] = None):
        data = self.data.copy()
        if features:
            data = data.loc[:, features]
        # data = pd.get_dummies(data=data)
        hdbscan_model = HDBSCAN(min_cluster_size=8)
        labels = hdbscan_model.fit_predict(data)

        # Visualize the clusters
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.title('HDBSCAN Clustering Result')
        plt.show()   
    
if __name__ == "__main__":
    cluster = Cluster(data_dir="./Data/Q6/Bank_Customer.csv")
    cluster.dbscan_fit(features=["TransactionAmount (INR)"])
    cluster.hdbscan_fit(features=["TransactionAmount (INR)"])
    cluster.kmeans_fit(features=["TransactionAmount (INR)"])