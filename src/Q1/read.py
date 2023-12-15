import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Optional


class Q1Data(Dataset):
    def __init__(self,
                 *,
                 files_dir: Optional[str] = None,
                 random_state: Optional[int] = 42,
                 train: Optional[bool] = True,
                 flag: Optional[bool] = True,
                 test: Optional[bool] = False,
                 test_file: Optional[str] = None,
                 align: Optional[bool] = None,
                 max_pairs: Optional[int] = None,
                 an_ratio: Optional[float] = 0.6) -> None:
        self.files_dir = files_dir
        self.random_state = random_state
        self.train = train
        self.flag = flag
        self.test = test
        self.align = align
        self.max_pairs = max_pairs
        self.an_ratio = an_ratio

        if self.train:
            if os.path.exists(files_dir):
                files = os.listdir(files_dir)
                file_list = []
                for file in files:
                    file_list.append(pd.read_csv(files_dir + file))
                df = pd.concat(file_list, axis=0)
                print(f"Total Number of training Sample: {len(df)}")

                if df.isna().any().any():
                    df.dropna(axis=0) # Drop the rows contain Nan

                data = df.iloc[:, :-1]
                label = df.iloc[:, -1]

                self.train_data, self.valid_data, self.train_label, self.valid_label \
                    = train_test_split(data, label, test_size=0.3, random_state=42)
                
                if self.align:
                    if self.max_pairs is not None:
                        self.pair_index, self.pair_label = self.align_pair(max_pairs=self.max_pairs)
                    else:
                        self.pair_index, self.pair_label = self.align_pair(max_pairs=3000)
            else:
                print(files_dir)
                raise FileNotFoundError("File path not exists")

        if self.test:
            if os.path.exists(test_file):
                self.test_data = pd.read_csv(test_file)
            else:
                raise FileNotFoundError

        
    def align_pair(self, max_pairs: Optional[int] = None):
        normal_index = self.train_label[self.train_label == 0].index
        anomaly_index = self.train_label[self.train_label == 1].index
        if max_pairs is None:
            nn_pair_size = int((1-self.an_ratio) * self.max_pairs)
            an_pair_size = int(self.an_ratio * self.max_pairs)
            nn_pair = [list(np.random.choice(normal_index.to_numpy(), size=nn_pair_size)), \
                        list(np.random.choice(normal_index.to_numpy(), size=nn_pair_size))]
            an_pair = [list(np.random.choice(anomaly_index.to_numpy(), size=an_pair_size)), \
                        list(np.random.choice(normal_index.to_numpy(), size=an_pair_size))]
        else:
            if max_pairs <= 2000:
                print("pairs should be greater than 2000, switch to default value 3000")
                max_pairs = self.max_pairs
            nn_pair_size = int((1-self.an_ratio) * max_pairs)
            an_pair_size = int(self.an_ratio * max_pairs)
            nn_pair = [list(np.random.choice(normal_index.to_numpy(), size=nn_pair_size)), \
                        list(np.random.choice(normal_index.to_numpy(), size=nn_pair_size))]
            an_pair = [list(np.random.choice(anomaly_index.to_numpy(), size=an_pair_size)), \
                        list(np.random.choice(normal_index.to_numpy(), size=an_pair_size))]

        self.normal_index = normal_index
        self.anomaly_index = anomaly_index
        pair = [nn_pair[0] + an_pair[0], nn_pair[1] + an_pair[1]]
        label = [1 for _ in range(len(nn_pair[0]))] + [-1 for _ in range(len(an_pair[0]))]
        return pair, label
    
    @property
    def train_anomaly_ratio(self):
        return len(self.train_label[self.train_label == 1]) / len(self.train_label)
    

    def __len__(self):
        if self.align:
            return self.max_pairs
        if self.train and self.flag:
            return len(self.train_label)
        else:
            if self.test:
                return len(self.test_data)
            else:
                return len(self.valid_label)
        
    def __getitem__(self, index):
        if not self.align:
            if self.train:
                data = torch.tensor(self.train_data.iloc[index, :].to_numpy(), dtype=torch.float32)
                label = torch.tensor(self.train_label.iloc[index, :].to_numpy())
            else:
                data = torch.tensor(self.valid_data.iloc[index, :].to_numpy(), dtype=torch.float32)
                label = torch.tensor(self.valid_label.iloc[index, :].to_numpy())
            return data, label 
        else:
            if self.train and self.flag:
                left = self.pair_index[0]
                right = self.pair_index[1]
                df = pd.DataFrame({
                    "left_index": left,
                    "right_index": right,
                    "label": self.pair_label
                })
                df.reset_index()
                batch_df = df.iloc[index, :]
                left_data = torch.tensor(self.train_data.iloc[batch_df["left_index"], :].to_numpy(), dtype=torch.float32)
                right_data = torch.tensor(self.train_data.iloc[batch_df["right_index"], :].to_numpy(), dtype=torch.float32)
                label = batch_df["label"]
                return left_data, right_data, label
            else:
                normal_data = torch.tensor(self.train_data.iloc[self.normal_index, :].to_numpy(), dtype=torch.float32)
                if self.test:
                    if isinstance(self.test_data, pd.DataFrame):
                        test_data = self.test_data.iloc[:, 1:-1] # move id column
                        test_label = self.test_data.iloc[:, -1]
                        test_data = torch.tensor(test_data.to_numpy(), dtype=torch.float32)
                        test_label = torch.tensor(test_label.to_numpy())
                        return test_data[index], normal_data[index], test_label[index]

                    else:
                        raise RuntimeError("Please set test data first")
                else:
                    valid_data = torch.tensor(self.valid_data.to_numpy(), dtype=torch.float32)
                    valid_label = torch.tensor(self.valid_label.to_numpy())
                    return valid_data[index], normal_data[index], valid_label[index]