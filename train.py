import torch
from dataloader import CTScanData
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import ViTForImageClassification, ViTConfig
import pandas as pd
import numpy as np

torch.manual_seed(1337)
print('Reading Data...')
df = pd.read_excel('./Data/image_data.xlsx')
dataset = CTScanData(df)
df = df.sample(frac=1).reset_index(drop=True)
print('Data reading success!')
split_ratio = 0.9
split = int(len(df)*split_ratio)
indices = [*range(len(df))]

train_sampler = SubsetRandomSampler(indices[:split])
valid_sampler = SubsetRandomSampler(indices[split:])

batch_size = 4
print('Preparing Dataloader')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
print('DataLoader successful')
print(f'{len(train_sampler)} images in Train loader | {len(valid_sampler)} images in Valid loader')
print(f'Num of train batches: {len(train_loader)} | Num of valid batches: {len(validation_loader)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is set to {device}')


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
    hidden_dropout_prob = 0.2,
    hidden_act = "gelu"

)

# Initialize the model
model = ViTForImageClassification(config)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

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
num_epochs = 5
print('Starting Training...')
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = validate(model, validation_loader, criterion, device)
    
    scheduler.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | '
          f'Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_acc:.4f}')
