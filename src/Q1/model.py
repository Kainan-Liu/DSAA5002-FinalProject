import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, Optional
from read import Q1Data
from sklearn.metrics import recall_score, accuracy_score, precision_score

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, act: Optional[bool] = True, last: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.last = last

        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True if not self.last else False),
            nn.BatchNorm1d(out_features) if not last else nn.Identity(),
            nn.Tanh() if act else nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)
    

class ANNet(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(ANNet, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        hidden_features = [16, 32, 64]
        self.model = nn.Sequential(
            LinearBlock(in_features=in_features, out_features=hidden_features[0]),
            LinearBlock(in_features=hidden_features[0], out_features=hidden_features[1]),
            LinearBlock(in_features=hidden_features[1], out_features=hidden_features[2]),
            LinearBlock(in_features=hidden_features[2], out_features=out_features, act=False, last=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.out_features, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def pretrain(self, lr, batch_size, epochs, max_sample):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = Q1Data(files_dir="c:/Users/lenovo/Desktop/HKUSTGZ-PG/Course-project/DSAA-5002/Final-Project/Data/Q1/train/", \
                        align=False, flag=True, max_sample=max_sample, pretrain=True, test=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        criterion = nn.BCELoss()
        model_parameters = self.model.parameters()
        classifier_parameters = self.classifier.parameters()
        all_parameters = itertools.chain(model_parameters, classifier_parameters)
        optimizer = optim.Adam(all_parameters, lr=lr, weight_decay=0.9)
        self.model.to(device=device)
        self.classifier.to(device=device)

        for epoch in range(epochs):
            t = tqdm(dataloader, leave=False, total=len(dataloader))
            t.set_description(f"Training Epoch: {epoch + 1}/{epochs}")
            for data, label in t:
                data = data.to(device=device)
                label = label.to(device=device)

                # forward
                output = self.model(data)
                pred = self.classifier(output).flatten()

                # backward
                loss = criterion(pred, label)
                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(Loss=loss.item())

        print("Test Pretrain model")
        self.model.eval()
        self.classifier.eval()
        test_data = pd.read_csv("c:/Users/lenovo/Desktop/HKUSTGZ-PG/Course-project/DSAA-5002/Final-Project/Data/Q1/test/test_set.csv")
        X = test_data.iloc[:, 1:-1]
        id = test_data.iloc[:, 0].tolist()
        y = test_data.iloc[:, -1]
        pred_labels = []
        test_labels = []
        with torch.no_grad():
            X = torch.tensor(X.to_numpy(), dtype=torch.float32, device=device)
            output = self.model(X)
            pred_label = torch.where(self.classifier(output).flatten() > 0.5, 1, 0).tolist()
            pred_labels += pred_label
            test_labels += y.tolist()
        
        df = pd.DataFrame({
            "ID": id,
            "Is_Falling": pred_labels
        })
        df.to_csv("./Data/Q1/Q1_output_upsample.csv", index=False)
        print("==Save Results==")
        print("Success!")
        
        print("======================Score=========================")
        print(f"Recall: {recall_score(test_labels, pred_labels, pos_label=1)}")
        print(f"Precision: {precision_score(test_labels, pred_labels, pos_label=1)}")
        print(f"Accuracy: {accuracy_score(test_labels, pred_labels)}")