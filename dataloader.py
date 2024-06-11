import torch
import torchvision.transforms as transforms
import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd


class CTScanData(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.labelencoder = LabelEncoder().fit(list(df['target'].unique()))
        # self.data_transfomers = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        image_path = Path(self.df.loc[index, 'image'])
        image = nib.load(image_path).get_fdata()
        image = torch.from_numpy(image)
        image = torch.permute(image, (2,0,1))
        image = image.unsqueeze(0).unsqueeze(0)
        # print(f'Original Shape: {image.shape}')
        # original_shape = image.shape
        target_shape = (224, 224, 224)
        image = torch.nn.functional.interpolate(image, size=target_shape, mode='trilinear')
        image = image.squeeze(0).squeeze(0)
        # print(f'After interpolation: {image.shape}')
        # print(self.df.loc[index, 'target'])
        label = torch.tensor(self.df.loc[index, 'target'])
    
        return image, label




if __name__ == "__main__":
    df = pd.read_excel('./Data/image_data.xlsx')
    dataset = CTScanData(df)
    print(dataset[1])



