import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
        self.target_size = (256, 256)  # target size for both image and labels

        # Image files
        self.image_files = sorted(glob(os.path.join(root_dir, '*.jpg')))
        if not self.image_files:
            raise ValueError(f"No .jpg files found in {root_dir}")

        # Lists to store valid pairs
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
        segmentation = segmentation.unsqueeze(0)  # Add channel dimension
        segmentation = TF.resize(segmentation, self.target_size, interpolation=TF.InterpolationMode.NEAREST)
        segmentation = segmentation.squeeze(0).long()  # Remove channel dimension and convert to long

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        return image, segmentation

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self._make_layer(3, 64)
        self.enc2 = self._make_layer(64, 128)
        self.enc3 = self._make_layer(128, 256)
        self.enc4 = self._make_layer(256, 512)

        # Decoder
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
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        out = self.dec1(d2)

        return out

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

def train_model(model, train_loader, val_loader, device, num_epochs=5, num_classes=1):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_acc = 0
        train_iou = np.zeros(num_classes)
        train_samples = 0

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
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
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

        # Epoch Statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
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

        # Optionally, save checkpoints every few epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

def test_model(model, test_loader, device, num_classes=1):
    model.eval()
    test_loss = 0
    test_acc = 0
    test_iou = np.zeros(num_classes)
    test_samples = 0

    criterion = nn.NLLLoss()

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for images, labels in test_pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # Calculate metrics
            _, preds = torch.max(outputs, 1)
            acc = calculate_pixel_accuracy(preds, labels)
            ious = calculate_iou(preds, labels, num_classes)

            test_acc += acc
            test_iou += np.nan_to_num(ious)
            test_samples += 1

            test_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / test_samples
    avg_test_iou = test_iou / test_samples

    print('\nTest Results:')
    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}, Test IoU: {avg_test_iou}')

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
    full_dataset = COCOStuffDataset(
        root_dir='data/images',
        annotations_dir='data/annotations',
        transform=transform
    )

    # Determine number of classes
    all_labels = []
    for _, label in tqdm(full_dataset, desc="Collecting labels for class determination"):
        all_labels.extend(torch.unique(label).tolist())
    unique_labels = sorted(set(all_labels))
    print(f"Unique labels in dataset: {unique_labels}")  # Debugging step
    n_classes = max(unique_labels) + 1  # Ensure all labels are included
    print(f"Number of unique classes: {n_classes}")

    # Split dataset into train, validation, and test
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))
    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")

    # Create data loaders
    batch_size = 8
    num_workers = 4  # Adjust based on your system

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize the model
    model = UNet(n_classes=n_classes).to(device)

    # Train the model
    num_epochs = 5 # Adjust as needed
    train_model(model, train_loader, val_loader, device, num_epochs=num_epochs, num_classes=n_classes)

    # Load the best model for testing
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded the best model for testing.")

    # Test the model
    test_model(model, test_loader, device, num_classes=n_classes)

if __name__ == "__main__":
    main()
