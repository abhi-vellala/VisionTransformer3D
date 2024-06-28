import nibabel as nib
import pandas as pd
import scipy.ndimage
import os
import numpy as np

df = pd.read_excel('./Data/image_data.xlsx')



def resample_image(image, new_shape):
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

new_shape = (224, 224, 224)

for idx, row in df[:1].iterrows():
    img_path = row['image']
    # seg_path = row['seg']
    img = nib.load(img_path)
    # seg = nib.load(seg_path)
    resized_img = resample_image(img.get_fdata(), new_shape)
    # resized_seg = resample_image(seg.get_fdata(), new_shape)
    final_img = nib.Nifti1Image(resized_img, img.affine)
    nib.save(final_img, os.path.join('./Data/', f'{row['ID']}_resampled_img.nii.gz'))
    # final_seg = nib.Nifti1Image(resized_seg, seg.affine)
    # nib.save(final_seg, os.path.join('./Data/Resampled/', f'{row['ID']}_resampled_seg.nii.gz'))