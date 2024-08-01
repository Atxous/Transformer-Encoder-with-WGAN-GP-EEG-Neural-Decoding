import torch
import os
import torchvision
import pickle
from torch.utils.data import Dataset

class char74kDataset(Dataset):
    def __init__(self, root : str, imgs_only = False, transforms = None, target_transforms = None):
        self.root = root
        self.classes = os.listdir(root)
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.imgs_only = imgs_only
        
    def __len__(self):
        return len(self.classes) * 1016
    
    def __getitem__(self, idx):
        class_num = idx // 1016
        img_num = idx % 1016
        
        img_root = os.path.join(self.root, self.classes[class_num])
        img = torchvision.io.read_image(os.path.join(img_root, os.listdir(img_root)[img_num]))
        if self.transforms:
            img = self.transforms(img)
        if self.imgs_only:
            return img
        
        target = torch.zeros(len(self.classes))
        target[class_num] = 1
        target = target.float()
        if self.target_transforms:
            target = self.target_transforms(target)
        
        return img, target
    
    def load_all_imgs(self):
        class_dirs = [os.path.join(self.root, class_dir) for class_dir in self.classes]
        imgs = []
        for class_dir in class_dirs:
            for img in os.listdir(class_dir):
                img = torchvision.io.read_image(os.path.join(class_dir, img))
                if self.transforms:
                    img = self.transforms(img)
                imgs.append(img)
        return imgs
    
class EEGClassificationDataset(Dataset):
    def __init__(self, root : str, train = True, transforms = None, target_transforms = None):
        with open(root, 'rb') as f:
            self.data = pickle.load(f, fix_imports=True, encoding='latin1')
        if train:
            self.targets = self.data["y_train"]
            self.data = self.data["x_train"].squeeze()
            
        else:
            self.targets = self.data["y_test"]
            self.data = self.data["x_test"].squeeze()
        
        self.transforms = transforms
        self.target_transforms = target_transforms
            
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        
        if self.transforms:
            data = self.transforms(data)
        if self.target_transforms:
            target = self.target_transforms(target)
        
        return data.float(), torch.tensor(target, dtype = torch.float32)
    
# class EEGDataset(Dataset):
#     def __init__(self, data, root : str, avg_repetitions = True, standardize_fn = None, transforms = None, target_transforms = None):
#         self.data = torch.tensor(data["preprocessed_eeg_data"]).float()
#         # Reshape from [Conditions, Repetitions, Channels, Time]
#         # to [Conditions, Repetitions, Time, Channels]
#         self.data = self.data.view(self.data.shape[0], self.data.shape[1], self.data.shape[3], self.data.shape[2])
#         # 1640 folders with different images
#         self.root = root
#         self.imgs = os.listdir(root)
#         self.transforms = transforms
#         self.target_transforms = target_transforms
        
#         # since the data deals with EEG repetitions, we can either average the data or split the data
#         # based on the different repetitions
#         self.avg_repetitions = avg_repetitions
#         if avg_repetitions:
#             self.data = torch.mean(self.data, dim=1)
#         else:
#             split_eeg_data = torch.split(self.data, self.data.shape[1], dim=1)
#             split_eeg_data = [torch.squeeze(split, dim=1) for split in split_eeg_data]
#             self.data = torch.stack(split_eeg_data, dim=0)
            
#         if standardize_fn:
#             self.data = standardize_fn(self.data)
        
#     def __len__(self):
#         # each folder contains 10 images
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         # Since range is from 0-16399, we can get the folder and image number by dividing and getting the remainder
#         folder = idx // 10
#         img_num = idx % 10
#         eeg_data = self.data[idx]
#         target_root = os.path.join(self.root, self.imgs[folder])
#         target = torchvision.io.read_image(os.path.join(target_root, os.listdir(target_root)[img_num]))
        
#         if self.transforms:
#             eeg_data = self.transforms(eeg_data)
#         if self.target_transforms:
#             target = self.target_transforms(target)
        
#         return eeg_data, target