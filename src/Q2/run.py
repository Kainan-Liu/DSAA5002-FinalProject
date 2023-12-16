import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import MyResNet
from read import Q2Dataset
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def main(num_classes, epochs, batch_size, lr):
    # initialization
    dataset = Q2Dataset(data_dir="../../Data/Q2/train_data/")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = MyResNet(num_classes=num_classes).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.9)

    print("======================================================")
    print("Begin Fine-tuning on Our weather dataset")
    t = tqdm(range(epochs), leave=False, total=epochs)
    for epoch in t:
        t.set_description("Training")
        for data, label in dataloader:
            data = data.to(device=device)
            label = label.to(device=device)

            # forward
            pred = model(data)

            # backward
            loss = criterion(pred, label)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(Loss=loss.item())
    print("End Training")
    print("======================================================")
    print("Begin Testing")
    test_dataset = Q2Dataset(data_dir="../../Data/Q2/train_data/", \
                             train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    model.eval()
    with torch.no_grad():
        pred_labels = []
        test_labels = []
        for test_data, test_label in test_dataloader:
            test_data = test_data.to(device=device)

            output = model(test_data)
            pred_label = F.softmax(output, dim=1).detach()
            pred_label = torch.argmax(pred_label, dim=1).tolist()
            test_label = torch.where(test_label > 0.1)[1].tolist()

            pred_labels += pred_label
            test_labels += test_label
    print(f"Accuracy: {accuracy_score(test_labels, pred_labels)}")


if __name__ == "__main__":
    seed_everything(random_state=42)
    main(num_classes=5, epochs=50, batch_size=16, lr=3e-5)