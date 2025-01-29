'''
DeepLabV3 Model:
DeepLabV3 is a neural network model for semantic segmentation.
It includes an encoder, an Atrous Spatial Pyramid Pooling (ASPP) module, and a decoder.
The forward method processes the input through these components and upsamples the output.

Loss functions:
Three loss functions are defined: Negative Log-Likelihood (NLL), Dice Loss, and Focal Loss.

Cross validation:
cross_validate performs k-fold cross-validation on the dataset.
It trains the model on different folds and logs the results.

Main functions:
Defines base configuration parameters and loops over different optimizer and loss function combinations.
'''



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms
import wandb
from sklearn.model_selection import KFold
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
        
        # loading image
        image = Image.open(img_path).convert('RGB')
        
        # loading annotation
        annotation = sio.loadmat(ann_path)
        segmentation = annotation['regionLabelsStuff']
        
        # convert segmentation to good format and resize
        segmentation = torch.from_numpy(segmentation).float()
        segmentation = segmentation.view(1, segmentation.size(0), -1)  # Add channel dimension
        segmentation = TF.resize(segmentation, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        segmentation = segmentation.squeeze(0).long()  # Remove channel dimension and convert to long
        
        # apply to image 
        if self.transform:
            image = self.transform(image)
        
        return image, segmentation

# DeepLabV3 model
class DeepLabV3(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._make_layer(3, 64, stride=1),
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2)
        )
        
        # ASPP
        self.aspp = nn.ModuleList([
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(512, 256, 3, padding=6, dilation=6),
            nn.Conv2d(512, 256, 3, padding=12, dilation=12),
            nn.Conv2d(512, 256, 3, padding=18, dilation=18)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1),
            nn.LogSoftmax(dim=1)
        )
        
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        
        aspp_out = [aspp_module(features) for aspp_module in self.aspp]
        aspp_out = torch.cat(aspp_out, dim=1)
        
        out = self.decoder(aspp_out)
        return self.upsample(out)

# loss functions
def get_loss_function(loss_name):
    losses = {
        'nll': nn.NLLLoss(),
        'dice': DiceLoss(),
        'focal': FocalLoss()
    }
    return losses[loss_name]

class DiceLoss(nn.Module):
    def forward(self, predictions, targets):
        smooth = 1.0
        predictions = torch.exp(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        intersection = (predictions * targets).sum()
        return 1 - ((2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(reduction='none')
    
    def forward(self, predictions, targets):
        nll_loss = self.nll(predictions, targets)
        probs = torch.exp(-nll_loss)
        focal_loss = ((1 - probs) ** self.gamma) * nll_loss
        return focal_loss.mean()

def train_model(model, train_loader, device, config):
    wandb.init(project="semantic-segmentation", config=config)
    
    optimizers = {
        'adam': optim.Adam(model.parameters(), lr=config['learning_rate']),
        'sgd': optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9),
        'adamw': optim.AdamW(model.parameters(), lr=config['learning_rate'])
    }
    optimizer = optimizers[config['optimizer']]
    
    criterion = get_loss_function(config['loss_function'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}')
    
    wandb.finish()

def cross_validate(dataset, device, config, k_folds=5):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=SubsetRandomSampler(train_ids), num_workers=4)
        
        model = DeepLabV3(config['n_classes']).to(device)
        fold_config = config.copy()
        fold_config['fold'] = fold
        
        train_model(model, train_loader, device, fold_config)

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = COCOStuffDataset(root_dir='data/images', annotations_dir='data/annotations', transform=transform)
    
    all_labels = []
    for _, label in dataset:
        all_labels.extend(torch.unique(label).tolist())
    n_classes = max(set(all_labels)) + 1
    
    base_config = {
        'n_classes': n_classes,
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 5
    }
    
    for optimizer_name in ['adam', 'sgd', 'adamw']:
        for loss_function in ['nll', 'dice', 'focal']:
            config = base_config.copy()
            config.update({'optimizer': optimizer_name, 'loss_function': loss_function})
            cross_validate(dataset, device, config)

if __name__ == "__main__":
    main()
