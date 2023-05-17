from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import os
import torchvision.transforms as transforms
from base_dataset import BaseDataset, get_transform, get_transform_lab, no_transform

# class AVADataset(Dataset):
#     def __init__(self, path_to_csv: Path,images_path : Path, transform=None):
#         self.df = pd.read_csv(path_to_csv)
#         self.images_path = images_path
#         self.transform = transform

#     def __len__(self) -> int:              #__len__(self)和__getitem__接口，实现一个索引/key到样本数据的map。
#         return self.df.shape[0]             #如 datasets[10]表示第十个样本

#     def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
#         row = self.df.iloc[item]

#         id = row["id"]
#         #print(id)
#         image_path = os.path.join(self.images_path , f"{id}.jpg")
#         image = default_loader(image_path)
#         x = self.transform(image)
       

#         y = row[1:].values.astype("float32")
#         p = y / y.sum()
#         # print('id1:',id,'label:',p)

#         return x,p
        
# class AVADataset(Dataset):
#     def __init__(self, path_to_csv: Path,images_path : Path, transform=None):
#         self.df = pd.read_csv(path_to_csv)
#         self.images_path = images_path
#         self.transform = transform

#     def __len__(self) -> int:              #__len__(self)和__getitem__接口，实现一个索引/key到样本数据的map。
#         return self.df.shape[0]             #如 datasets[10]表示第十个样本

#     def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
#         row = self.df.iloc[item]

#         id = row["id"]
#         #print(id)
#         image_path = os.path.join(self.images_path , f"{id}.jpg")
#         image = default_loader(image_path)
#         x = self.transform(image)


#         # # df=pd.read_csv("train.csv")????????
#         score = 0
#         titles_to_dataset = ["id", "score1", "score2", "score3", "score4", "score5", "score6", "score7","score8",
#                              "score9", "score10"]
#         for i in range(10):
#             score +=  row[titles_to_dataset[i + 1]] * (i + 1)

#         y = row[1:].values.astype("float32")
#         person = y.sum()
#         # print(person)
#         ava_score =score / person
#         # print(ava_score)
#         # label=round(ava_score,2).astype("float32")
#         # label=ava_score
#         # 得到标签
#         # label = ava_score.apply(lambda x: 1 if x > 5 else 0)
#         # if ava_score>5:
#         #     label=1
#         # else:
#         #     label=0
#         # print(label)
#         #将id和label连接在一起
    
#         return x,ava_score


###############################融合：将图像转换为LAB空间，返回图像，标签，LAB图像

class AVADataset(Dataset):
    def __init__(self, path_to_csv: Path,images_path : Path, transform=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.transform = transform
        self.transform_type= get_transform_lab()
    def __len__(self) -> int:              #__len__(self)和__getitem__接口，实现一个索引/key到样本数据的map。
        return self.df.shape[0]             #如 datasets[10]表示第十个样本

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        row = self.df.iloc[item]

        id = row["id"]
        #print(id)
        image_path = os.path.join(self.images_path , f"{id}.jpg")
        image = default_loader(image_path)
        x = self.transform(image)


        # # df=pd.read_csv("train.csv")????????
        score = 0
        titles_to_dataset = ["id", "score1", "score2", "score3", "score4", "score5", "score6", "score7","score8",
                             "score9", "score10"]
        for i in range(10):
            score +=  row[titles_to_dataset[i + 1]] * (i + 1)

        y = row[1:].values.astype("float32")
        person = y.sum()
        # print(person)
        ava_score =score / person
        img=image.resize(size=(224, 224))
        lab = self.transform_type(img)
    
        return x,ava_score,lab 