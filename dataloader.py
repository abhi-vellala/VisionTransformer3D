# Pre-DataLoader for model training. 
# Encodes target into labels
# Reads images and gives tensors
# Resample function for image interpolation
# gives DataLoader with images and labels

import torch
# import torchvision.transforms as transforms
import os
import numpy as np
import nibabel as nib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import scipy.ndimage
from sklearn.preprocessing import StandardScaler


class CTScanData(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.labelencoder = LabelEncoder().fit(list(df['target'].unique()))
        # self.data_transfomers = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

    def resample_image(self, image, new_shape):
        original_shape = image.shape
        
        # Calculate the resize factor for each dimension
        resize_factor = [
            new_shape[0] / original_shape[0],
            new_shape[1] / original_shape[1],
            new_shape[2] / original_shape[2]
        ]
        
        # Perform the resampling
        resized_image = scipy.ndimage.zoom(image, resize_factor, order=1)  # order=1 for linear interpolation
        return np.round(resized_image, 0)
    
    def normalize(self, image):
        scaler = StandardScaler()
        image = image.astype(np.float32)
        image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
        return image
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        image_path = Path(self.df.loc[index, 'image'])
        image = nib.load(image_path).get_fdata()
        # image = self.normalize(image)
        image = torch.from_numpy(image)
        target_shape = (224, 224, 224)
        image = self.resample_image(image, target_shape)
        image = torch.tensor(image).unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        image = image.squeeze(0)
        label = torch.tensor(self.df.loc[index, 'target'])
        
        return image, label




if __name__ == "__main__":
    df = pd.read_excel('./Data/image_data.xlsx')
    dataset = CTScanData(df)
    # print(dataset[1][0].size())
    img = dataset[1][0].numpy()
    print(img)
    # img_nib = nib.Nifti1Image(img, affine=np.eye(4))
    # nib.save(img_nib, './Data/1_img_interpolated.nii.gz')




