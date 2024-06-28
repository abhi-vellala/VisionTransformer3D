import torch
from dataloader import CTScanData
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import ViTForImageClassification, ViTConfig
import pandas as pd
import os
import time
from vit_model import ViTModelCustom
import torchio as tio
from sklearn.model_selection import train_test_split

torch.manual_seed(1234567890)
print('Reading Data...')
df = pd.read_excel('./Data/image_data.xlsx')
dataset = CTScanData(df)
df = df.sample(frac=1).reset_index(drop=True)
print('Data reading success!')
# split_ratio = 0.8
# split = int(len(df)*split_ratio)
# indices = [*range(len(df))]

# train_sampler = SubsetRandomSampler(indices[:split])
# valid_sampler = SubsetRandomSampler(indices[split:])

train_set, valid_set = train_test_split(df, test_size=0.2, stratify=df['target'])
train_sampler = list(train_set.index)
valid_sampler = list(valid_set.index)

batch_size = 4
print('Preparing Dataloader')

train_transforms = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    tio.RandomFlip(axes=(0, 1, 2)),
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad((224, 224, 224)),
    tio.ToCanonical()
])

validation_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.CropOrPad((224, 224, 224)),
    tio.ToCanonical()
])

train_dataset = CTScanData(df.iloc[train_sampler].reset_index(drop=True), transforms_=train_transforms)
validation_dataset = CTScanData(df.iloc[valid_sampler].reset_index(drop=True))
batch_size = 4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
print('DataLoader successful')
print(f'{len(train_sampler)} images in Train loader | {len(valid_sampler)} images in Valid loader')
print(f'Num of train batches: {len(train_loader)} | Num of valid batches: {len(validation_loader)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is set to {device}')

model_results = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
model_save_path = './Data/model_save/'

# Configure the Vision Transformer
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=224,  # Depth dimension as channels
    num_labels=2,     # Number of classes
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob = 0.1,
    hidden_act = "gelu"

)

# Initialize the model
model = ViTForImageClassification(config)
# model = ViTModelCustom(config, device)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Training function
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

# Validation function
def validate(model, validation_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    valid_loss = running_loss / len(validation_loader)
    valid_acc = correct / total
    return valid_loss, valid_acc

# Training loop
num_epochs = 550
least_val_loss = 0
print('Starting Training...')
start = time.time()
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)    
    valid_loss, valid_acc = validate(model, validation_loader, criterion, device)
    model_results.loc[epoch, 'epoch'] = epoch +1
    model_results.loc[epoch, 'train_loss'] = train_loss
    model_results.loc[epoch, 'valid_loss'] = valid_loss
    model_results.loc[epoch, 'train_acc'] = train_acc
    model_results.loc[epoch, 'valid_acc'] = valid_acc
    # scheduler.step()
    if num_epochs > 0 and valid_loss < least_val_loss:    
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pt'))
        least_val_loss = valid_loss
        print(f'Model saved at: {epoch+1}')
    else:
        least_val_loss = valid_loss
    

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | '
          f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}')

model_results.to_excel('./Data/results.xlsx', index=False)

end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Execution Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))