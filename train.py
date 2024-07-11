import torch
from dataloader import CTScanData
import pandas as pd
import os
import time
from vit_model import VisionTransformer3D
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, RandRotate90, RandFlip,
    RandZoom, RandGaussianNoise, SpatialPad, RandAffine, ToNumpy, ToTensor
)

device_type = 'local' # server

batch_size = 1 if device_type == 'local' else 16
in_channels = 1
d_model = 256
feedforward_dim = 512
num_heads = 8
patch_size = 16
num_layers = 1
dropout = 0.1
num_classes = 2
model_save_path = './Data/model_save/'
test_size = 0.25
learning_rate = 1e-4
weight_decay=0.01
num_epochs = 5

if device_type == 'server':
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(1234567890)
print('Reading Data...')
df = pd.read_excel('./Data/image_data.xlsx')
dataset = CTScanData(df)
df = df.sample(frac=1).reset_index(drop=True)
print('Data reading success!')

train_set, valid_set = train_test_split(df, test_size=test_size, stratify=df['target'])
train_sampler = list(train_set.index)
valid_sampler = list(valid_set.index)

print('Preparing Dataloader')

train_transforms = Compose([
        ScaleIntensity(),
        RandRotate90(prob=0.5),
        RandFlip(prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        RandGaussianNoise(prob=0.5),
        EnsureChannelFirst(channel_dim="no_channel"),
        # SpatialPad((224, 224, 224)),
        # RandAffine(prob=0.5),
        ToTensor()
    ])

val_transforms = Compose([
        ScaleIntensity(),
        EnsureChannelFirst(channel_dim="no_channel"),
        # SpatialPad((224, 224, 224)),
        ToTensor()
    ])

train_dataset = CTScanData(df.iloc[train_sampler].reset_index(drop=True), transform=train_transforms)
validation_dataset = CTScanData(df.iloc[valid_sampler].reset_index(drop=True), transform=val_transforms)

if device_type =='server':
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True
    )
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)


print('DataLoader successful')
print(f'{len(train_sampler)} images in Train loader | {len(valid_sampler)} images in Valid loader')
print(f'Num of train batches: {len(train_loader)} | Num of valid batches: {len(validation_loader)}')


print(f'Device is set to {device}')

model_results = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'])


model = VisionTransformer3D(in_channels, d_model, feedforward_dim, patch_size, num_classes=2, num_layers=num_layers)
if device_type == 'server':
     model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])
else:
    model = model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device, local_rank):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, attn_weights = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Reduce and aggregate the loss
        if device_type == 'server':
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
            running_loss += loss.item() if local_rank == 0 else 0
        else:
            running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    if device_type == 'server':
        # Compute average loss and accuracy on rank 0
        if local_rank == 0:
            train_loss = running_loss / (len(train_loader) * torch.distributed.get_world_size())
            train_acc = correct / total
        else:
            train_loss = None
            train_acc = None
    else:
        train_loss = running_loss / (len(train_loader))
        train_acc = correct / total

        
    return train_loss, train_acc

def validate(model, validation_loader, criterion, device, local_rank):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)
            
            if device_type == 'server':
                # Reduce and aggregate the loss
                torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.SUM)
                running_loss += loss.item() if local_rank == 0 else 0
            else:
                running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    if device_type == 'server':
        # Compute average loss and accuracy on rank 0
        if local_rank == 0:
            valid_loss = running_loss / (len(validation_loader) * torch.distributed.get_world_size())
            valid_acc = correct / total
        else:
            valid_loss = None
            valid_acc = None
    else:
        valid_loss = running_loss / (len(validation_loader))
        valid_acc = correct / total
        
    return valid_loss, valid_acc

least_val_loss = float('inf')
print('Starting Training...')
start = time.time()
for epoch in range(num_epochs):
    local_rank = local_rank if device_type == 'server' else None
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, local_rank)    
    valid_loss, valid_acc = validate(model, validation_loader, criterion, device, local_rank)
    scheduler.step()
    
    # Only log results on the main process
    if local_rank == 0:
        model_results.loc[epoch, 'epoch'] = epoch + 1
        model_results.loc[epoch, 'train_loss'] = train_loss
        model_results.loc[epoch, 'valid_loss'] = valid_loss
        model_results.loc[epoch, 'train_acc'] = train_acc
        model_results.loc[epoch, 'valid_acc'] = valid_acc
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}_model.pt'))
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | '
              f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}')
        if valid_loss < least_val_loss:    
            least_val_loss = valid_loss
            print(f'Least Valid Loss: {least_val_loss} at epoch: {epoch+1}')
        # else:
        #     least_val_loss = valid_loss

if local_rank == 0:
    model_results.to_excel('./Data/results.xlsx', index=False)

end = time.time()
if local_rank == 0:
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Execution Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))