import torch
import torch.nn as nn
import numpy as np
import copy
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from read import Q1Data
from model import ANNet
from sklearn.metrics import recall_score, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def main(out_features = 16,
         batch_size = 64,
         lr = 1e-4,
         epochs=1,
         max_sample=None,
         max_pairs=1000,
         an_ratio = 0.5,
         temperature = 0.8,
         pretrain_epochs=10,
         pretrain = True):

    # initialization
    dataset = Q1Data(files_dir="../../Data/Q1/train/", \
                     align=True, max_pairs=max_pairs, test=False, an_ratio=an_ratio)
    input_features = dataset.train_data.shape[1]

    if pretrain:
        print("Begin Pretrain")
        model = ANNet(in_features=input_features, out_features=out_features).to(device=device)
        model.pretrain(lr=1e-5, batch_size=256, epochs=pretrain_epochs, max_sample=max_sample)
        left_net = model
        right_net = copy.deepcopy(model)
    else:
        left_net = ANNet(in_features=input_features, out_features=out_features).to(device=device)
        right_net = ANNet(in_features=input_features, out_features=out_features).to(device=device)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    criterion = nn.L1Loss()

    left_optimizer = optim.Adam(left_net.parameters(), lr=lr, weight_decay=0.9)
    right_optimizer = optim.Adam(right_net.parameters(), lr=lr, weight_decay=0.9)


    print("Begin Anomaly Detection")
    for epoch in range(epochs):
        t = tqdm(dataloader, leave=False, total=len(dataloader))
        for left_data, right_data, label in t:
            t.set_description(f"Anomaly-Detection Training Epoch {epoch + 1}/{epochs}")

            left_data = left_data.to(device=device)
            right_data = right_data.to(device=device)
            label = label.to(device=device)

            # forward
            left_embedding = left_net(left_data)
            right_embedding = right_net(right_data) # batch_size * output_features
            left_embedding_norm = torch.norm(left_embedding, dim=1)
            right_embedding_norm = torch.norm(right_embedding, dim=1)
            pred = torch.diag(torch.mm(left_embedding, right_embedding.t())) \
                  / (left_embedding_norm * right_embedding_norm) * temperature

            # backward
            loss = criterion(pred, label)

            # update
            left_optimizer.zero_grad()
            right_optimizer.zero_grad()
            loss.backward()
            left_optimizer.step()
            right_optimizer.step()

            t.set_postfix(Loss=loss.item())
    print("End Training")
    return left_net, right_net

def test(left_net, right_net, threshold: float = -0.2):

    print("====================================================")
    print(f"Begin detect outliers on Test under {threshold}")
    test_dataset = Q1Data(test_file="../../Data/Q1/test/test_set.csv", \
                          align=True, max_pairs=3000, test=True, flag=False,
                          files_dir="../../Data/Q1/train/")
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    left_net.eval()
    right_net.eval()
    with torch.no_grad():
        loop = tqdm(test_dataloader, leave=False, total=len(test_dataloader))
        pred_labels = []
        test_labels = []
        for test_data, normal_data, test_label in loop:
            test_data = test_data.to(device=device)
            normal_data = normal_data.to(device=device)
            test_label = test_label.to(device=device)

            test_left_embedding = left_net(test_data)
            test_right_embedding = right_net(normal_data)
            left_embedding_norm = torch.norm(test_left_embedding, dim=1)
            right_embedding_norm = torch.norm(test_right_embedding, dim=1)
            pred = torch.diag(torch.mm(test_left_embedding, test_right_embedding.t())) \
                / (left_embedding_norm * right_embedding_norm)
            pred_label = torch.where(pred > threshold, 0, 1).tolist()
            pred_labels += pred_label
            test_labels += test_label
    
    print("======================Score=========================")
    print(f"Recall: {recall_score(test_labels, pred_labels, pos_label=1)}")
    print(f"Accuracy: {accuracy_score(test_labels, pred_labels)}")


if __name__ == "__main__":
    print("====================================================")
    seed_everything(random_state=42)
    left_net, right_net = main(out_features = 8,
                               batch_size = 128,
                               lr = 2e-4,
                               pretrain_epochs=10,
                               epochs=2,
                               max_pairs=6000,
                               an_ratio=0.4,
                               temperature=0.6)
    test(left_net=left_net, right_net=right_net, threshold=0.3)
    test(left_net=left_net, right_net=right_net, threshold=-0.1)
    test(left_net=left_net, right_net=right_net, threshold=0)
