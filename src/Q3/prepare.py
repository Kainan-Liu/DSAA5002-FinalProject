import os
import pandas as pd
from typing import Optional, Literal

def write_video_to_csv(file_path: Optional[str] = None,
                       mode: Literal["train", "test"] = "test"):
    assert file_path is not None, "Please provide the video folder"
    if os.path.exists(file_path):
        folder = mode + "_video"
        file_folder = os.path.join(file_path, folder)
        files = os.listdir(file_folder)
        df = pd.DataFrame(files, columns=["path"])
        df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: os.path.join(f"../../Data/Q3/{mode}_video/", x))
        df.to_csv(f"../../Data/Q3/{mode}_video_file.csv", index=False)
    else:
        raise FileNotFoundError
        

def write_video_label_to_csv(file_path: Optional[str]=None,
                             mode: Literal["train", "test"] = "train"):
    assert file_path is not None, "please provide the video_tag.txt file"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, names=["path", "label"], header=None)
        df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: os.path.join(f"../../Data/Q3/{mode}_video/", x))
        df.to_csv(f"../../Data/Q3/{mode}_video_file.csv", index=False)
    else:
        raise FileNotFoundError