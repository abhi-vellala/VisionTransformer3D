import torch
from dataloader import CTScanData
import pandas as pd

from sklearn.model_selection import train_test_split
# import torchvision
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, RandRotate90, RandFlip,
    RandZoom, RandGaussianNoise, SpatialPad, RandAffine, ToNumpy
)

torch.manual_seed(1234567890)
print('Reading Data...')
df = pd.read_excel('./Data/image_data.xlsx')
dataset = CTScanData(df)
df = df.sample(frac=1).reset_index(drop=True)
print('Data reading success!')

train_set, valid_set = train_test_split(df, test_size=0.2, stratify=df['target'])
train_sampler = list(train_set.index)
valid_sampler = list(valid_set.index)

print('Preparing Dataloader')

# train_transforms = tio.Compose([
#     tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
#     tio.RandomFlip(axes=(0, 1, 2)),
#     tio.RescaleIntensity(out_min_max=(0, 1)),
#     tio.CropOrPad((224, 224, 224)),
#     tio.ToCanonical()
# ])

# validation_transforms = tio.Compose([
#     tio.RescaleIntensity(out_min_max=(0, 1)),
#     tio.CropOrPad((224, 224, 224)),
#     tio.ToCanonical()
# ])

train_transforms = Compose([
        # LoadImage(image_only=True),
        ScaleIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        RandGaussianNoise(prob=0.5),
        SpatialPad((224, 224, 224)),
        RandAffine(prob=0.5),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy(),
        # torchvision.transforms.ToTensor(),
    ])

val_transforms = Compose([
        # LoadImage(image_only=True),
        ScaleIntensity(),
        SpatialPad((224, 224, 224)),
        EnsureChannelFirst(channel_dim="no_channel"),
        ToNumpy(),
        # torchvision.transforms.ToTensor(),
    ])

train_dataset = CTScanData(df.iloc[train_sampler].reset_index(drop=True), transform=train_transforms)
validation_dataset = CTScanData(df.iloc[valid_sampler].reset_index(drop=True), transform=val_transforms)
batch_size = 2

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=0)
print('DataLoader successful')
print(f'{len(train_sampler)} images in Train loader | {len(valid_sampler)} images in Valid loader')
print(f'Num of train batches: {len(train_loader)} | Num of valid batches: {len(validation_loader)}')

example = next(iter(train_loader))[0]
print(type(example))
print(example.size())
print(example.to('cuda'))