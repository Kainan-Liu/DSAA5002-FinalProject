import torchvision
import datasets
import transforms
import torch
import torch.nn as nn
import os
import pandas as pd
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from model import Swin3d_Wrapper
from prepare import write_video_to_csv, write_video_label_to_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(random_state):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)


def main(num_classes, epochs, batch_size, lr, test_file):
    # initialization
    write_video_label_to_csv(file_path="../../Data/Q3/train_tag.txt", mode="train")
    dataset = datasets.VideoLabelDataset(
        "../../Data/Q3/train_video_file.csv",
        transform=torchvision.transforms.Compose([
            transforms.VideoFilePathToTensor(max_len=50, fps=5, padding_mode='last'),
            transforms.VideoRandomCrop([400, 400]),
            transforms.VideoResize([256, 256], torchvision.transforms.InterpolationMode.BICUBIC),
        ])
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
    model = Swin3d_Wrapper(num_classes=num_classes).to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.9)

    print("======================================================")
    print("Begin Fine-tuning on Our weather dataset")
    for epoch in range(epochs):
        t = tqdm(dataloader, leave=False, total=len(dataloader))
        t.set_description("Training")
        for data, label in t:
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
    write_video_to_csv(file_path="../../Data/Q3/", mode="test")
    dataset = datasets.VideoDataset(
        "../../Data/Q3/test_video_file.csv",
        transform=torchvision.transforms.Compose([
            transforms.VideoFilePathToTensor(max_len=50, fps=5, padding_mode='last'),
            transforms.VideoRandomCrop([400, 400]),
            transforms.VideoResize([256, 256], torchvision.transforms.InterpolationMode.BICUBIC),
        ])
    )
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        pred_labels = []
        for test_data in test_dataloader:
            test_data = test_data.to(device=device)

            output = model(test_data)
            pred_label = F.softmax(output, dim=1).detach()
            pred_label = torch.argmax(pred_label, dim=1).tolist()

            pred_labels += pred_label
    files = os.listdir(test_file)
    submit_df = pd.DataFrame({
        "file_name": files,
        "label": pred_labels
    })
    submit_df.to_csv("../../Data/Q3/Q3_output.csv", index=False)
    print("==>Submit Successfully!")


if __name__ == "__main__":
    seed_everything(random_state=42)
    main(num_classes=15, epochs=1, batch_size=2, lr=2e-5)