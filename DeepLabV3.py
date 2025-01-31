import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
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
            raise ValueError(f"No .jpg files found in {root_dir}")

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
        
        image = Image.open(img_path).convert('RGB')
    
        annotation = sio.loadmat(ann_path)
        segmentation = annotation['regionLabelsStuff']

        segmentation = torch.from_numpy(segmentation).float()
        segmentation = segmentation.view(1, segmentation.size(0), -1)  # Add channel dimension
        segmentation = TF.resize(segmentation, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        segmentation = segmentation.squeeze(0).long()  # Remove channel dimension and convert to long
        
        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        
        return image, segmentation


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
        predictions = torch.exp(predictions)  # Convert log probabilities to probabilities
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

def calculate_iou(pred, target, num_classes):
    """Calculate Intersection over Union (IoU) for each class."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou = float('nan')  # To ignore this class in mean IoU
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def calculate_pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy."""
    correct = (pred == target).float()
    acc = correct.sum() / correct.numel()
    return acc.item()

def train_model(model, train_loader, val_loader, device, config, num_classes):
    wandb.init(project="semantic-segmentation", config=config)
    
    optimizers = {
        'adam': optim.Adam(model.parameters(), lr=config['learning_rate']),
        'sgd': optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9),
        'adamw': optim.AdamW(model.parameters(), lr=config['learning_rate'])
    }
    optimizer = optimizers[config['optimizer']]
    
    criterion = get_loss_function(config['loss_function'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Training Phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = np.zeros(num_classes)
        train_samples = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate metrics
            _, preds = torch.max(outputs, 1)
            acc = calculate_pixel_accuracy(preds, labels)
            ious = calculate_iou(preds, labels, num_classes)

            train_acc += acc
            # Handle NaN in IoU (ignore classes not present in the batch)
            train_iou += np.nan_to_num(ious)
            train_samples += 1

            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / train_samples
        avg_train_iou = train_iou / train_samples

        # Validation Phase
        model.eval()
        val_loss = 0
        val_acc = 0
        val_iou = np.zeros(num_classes)
        val_samples = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate metrics
                _, preds = torch.max(outputs, 1)
                acc = calculate_pixel_accuracy(preds, labels)
                ious = calculate_iou(preds, labels, num_classes)

                val_acc += acc
                val_iou += np.nan_to_num(ious)
                val_samples += 1

                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / val_samples
        avg_val_iou = val_iou / val_samples

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Logging to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "train_iou": avg_train_iou.tolist(),
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_iou": avg_val_iou.tolist(),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Epoch Statistics
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Acc: {avg_train_acc:.4f}, Training IoU: {avg_train_iou}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Acc: {avg_val_acc:.4f}, Validation IoU: {avg_val_iou}')

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, 'best_model.pth')
            print("Best model saved.")

    wandb.finish()


def cross_validate(dataset, device, config, k_folds=5, num_classes=1):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'\nFOLD {fold+1}')
        print('--------------------------------')
        
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        # Define data loaders for training and validation
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=train_subsampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=config['batch_size'], sampler=val_subsampler, num_workers=4)
        
        # Initialize the model
        model = DeepLabV3(config['n_classes']).to(device)
        
        # Pass the fold number to the config
        fold_config = config.copy()
        fold_config['fold'] = fold + 1
        
        # Train the model
        train_model(model, train_loader, val_loader, device, fold_config, num_classes)
        
        print('--------------------------------')

def main():
    # Device configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the dataset
    dataset = COCOStuffDataset(root_dir='data/images', annotations_dir='data/annotations', transform=transform)
    
    # Determine number of classes
    all_labels = []
    for _, label in tqdm(dataset, desc="Collecting labels for class determination"):
        all_labels.extend(torch.unique(label).tolist())
    unique_labels = sorted(set(all_labels))
    print(f"Unique labels in dataset: {unique_labels}")  # Debugging step
    n_classes = max(unique_labels) + 1  # Ensure all labels are included
    print(f"Number of unique classes: {n_classes}")
    
    # Update the configuration with the number of classes
    base_config = {
        'n_classes': n_classes,
        'batch_size': 8,
        'learning_rate': 0.001,
        'num_epochs': 5,
        'optimizer': 'adam',  # Can be 'adam', 'sgd', 'adamw'
        'loss_function': 'nll'  # Can be 'nll', 'dice', 'focal'
    }
    
    # Start cross-validation
    cross_validate(dataset, device, base_config, k_folds=5, num_classes=n_classes)

if __name__ == "__main__":
    main()
