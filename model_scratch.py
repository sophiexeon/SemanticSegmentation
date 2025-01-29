import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import os

class COCOStuffDataset(Dataset):
    def __init__(self, root_dir, annotations_dir, transform=None):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_size = (256, 256)  
        
        # get images 
        self.image_files = sorted(glob(os.path.join(root_dir, '*.jpg')))
        if not self.image_files:
            raise ValueError(f"No .png files found in {root_dir}")
        
        # lists to store valid pairs
        self.valid_pairs = []
        self._find_valid_pairs()
    
    def _find_valid_pairs(self):
        print("Finding matching image-annotation pairs...")
        for img_path in tqdm(self.image_files):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(self.annotations_dir, f"{base_name}.mat")
            
            if os.path.exists(ann_path):
                try:
                    annotation = sio.loadmat(ann_path)
                    if 'regionLabelsStuff' in annotation:
                        self.valid_pairs.append((img_path, ann_path))
                except Exception as e:
                    print(f"Error loading {ann_path}: {str(e)}")
        print(f"Found {len(self.valid_pairs)} valid image-annotation pairs")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, ann_path = self.valid_pairs[idx]
        
        # load image
        image = Image.open(img_path).convert('RGB')
        
        # load annotation
        annotation = sio.loadmat(ann_path)
        segmentation = annotation['regionLabelsStuff']
        
        # convert segmentation to correct format and resize
        segmentation = torch.from_numpy(segmentation).float()
        segmentation = segmentation.view(1, segmentation.size(0), -1)  # add channel dimension
        segmentation = TF.resize(segmentation, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        segmentation = segmentation.squeeze(0).long()  # remove channel dimension and convert to long
        
        # apply transforms to image
        if self.transform:
            image = self.transform(image)
        
        return image, segmentation

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        
        # encoder
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)
        
        # decoder
        self.dec4 = self._make_layer(512 + 256, 256)
        self.dec3 = self._make_layer(256 + 128, 128)
        self.dec2 = self._make_layer(128 + 64, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        out = self.dec1(d2)
        
        return out

def train_model(model, train_loader, val_loader, device, num_epochs=5):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # training 
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # epoch statistics
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        
        # save checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }, f'checkpoint_epoch_{epoch}.pth')

def main():
    # check if GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # create dataset instances
    full_dataset = COCOStuffDataset(
        root_dir='data/images',
        annotations_dir='data/annotations',
        transform=transform
    )
    
    # check number of classes
    all_labels = []
    for _, label in full_dataset:
        all_labels.extend(torch.unique(label).tolist())
    print(f"Unique labels in dataset: {sorted(set(all_labels))}")  # debugging 
    n_classes = max(set(all_labels)) + 1  # check if all labels are included
    print(f"Number of unique classes: {n_classes}")
    
    # split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # initialise model
    model = UNet(n_classes).to(device)
    
    # train model
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()