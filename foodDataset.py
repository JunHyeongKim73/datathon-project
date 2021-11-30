import pandas as pd
import numpy as np
import os
from PIL import Image
import cv2
import random
from torch.utils.data import Dataset
import time
    
class FoodDataset(Dataset):
    def __init__(self, label, root):
        self.img_dir = root 
        self.img_labels = pd.read_csv(label, names=['path', 'label'])
        self.num_per_class = 500
        
        self.indicies = self.getIndicies(self.num_per_class)
        self.img_labels = self.img_labels.iloc[self.indicies]
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image 불러오기
        img = cv2.imread(img_path)
        if img is None:
            print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # label 불러오기
        label = self.img_labels.iloc[idx, 1]
        
        return (img, label)
    
    def getIndicies(self, num_per_class):
        index_lists = []
        for label in range(150):
            lists = self.img_labels[self.img_labels['label'] == label].index.tolist()
            lists = random.sample(lists, num_per_class)
            index_lists.extend(lists)
        
        return index_lists
    
# class AllDataset(Dataset):
#     def __init__(self, label, root):
#         self.img_dir = root 
#         self.img_labels = pd.read_csv(label, names=['path', 'label'])
        
#     def __len__(self):
#         return len(self.img_labels)
    
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         # image 불러오기
#         img = cv2.imread(img_path)
#         if img is None:
#             print(img_path[7:])
#         # label 불러오기
#         label = self.img_labels.iloc[idx, 1]
        
#         return (img, label)