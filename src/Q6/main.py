from sklearn.cluster import KMeans, DBSCAN
from typing import Optional
import os
import pandas as pd

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
        data = pd.get_dummies(data=data)
        kmeans_model =KMeans(n_clusters=8, n_init="auto", random_state=42)
        kmeans_model.fit(data)
        predict_label = kmeans_model.labels_
        print(predict_label)

    def dbscan_fit(self, features: Optional[list] = None):
        data = self.data.copy()
        if features:
            data = data.loc[:, features]
        data = pd.get_dummies(data=data)
        dbscan_model = DBSCAN()
        dbscan_model.fit(data)
        predict_label = dbscan_model.labels_
        print(predict_label)

    # def hdbscan_fit(self, features: Optional[list] = None):
    #     data = self.data.copy()
    #     if features:
    #         data = data.loc[:, features]
    #     data = pd.get_dummies(data=data)
    #     HDBSCAN()       
    
if __name__ == "__main__":
    cluster = Cluster(data_dir="./Data/Q6/Bank_Customer.csv")
    cluster.kmeans_fit(features=["CustGender", "CustLocation"])