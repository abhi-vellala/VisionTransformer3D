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


class CTScanData(torch.utils.data.Dataset):
    def __init__(self, df, transforms_=None):
        self.df = df
        self.labelencoder = LabelEncoder().fit(list(df['target'].unique()))
        # self.data_transfomers = transforms.Compose([transforms.ToTensor()])
        self.transforms_ = transforms_

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
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        image_path = Path(self.df.loc[index, 'image'])
        image = nib.load(image_path).get_fdata()
        image = torch.from_numpy(image)
        target_shape = (224, 224, 224)
        image = self.resample_image(image, target_shape)
        image = torch.tensor(image).unsqueeze(0)
        # image = torch.permute(image, (2,0,1))
        # image = image.unsqueeze(0).unsqueeze(0)
        # print(f'Original Shape: {image.shape}')
        # original_shape = image.shape
        
        # image = torch.nn.functional.interpolate(image, size=target_shape, mode='trilinear')
        # image = image.squeeze(0).squeeze(0)
        # print(f'After interpolation: {image.shape}')
        # print(self.df.loc[index, 'target'])
        if self.transforms_:
            image = self.transforms_(image)
        image = image.squeeze(0)
        label = torch.tensor(self.df.loc[index, 'target'])
        
        return image, label




if __name__ == "__main__":
    df = pd.read_excel('./Data/image_data.xlsx')
    dataset = CTScanData(df)
    # print(dataset[1][0].size())
    img = dataset[1][0].numpy()
    print(img)
    img_nib = nib.Nifti1Image(img, affine=np.eye(4))
    nib.save(img_nib, './Data/1_img_interpolated.nii.gz')




