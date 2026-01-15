import os
import random
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR # [Option 2] Import Scheduler

# ===========================================================
# 1. Fix Random Seed (Ensure Reproducibility)
# ===========================================================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed fixed to: {seed}")

setup_seed(42)

# ===========================================================
# 2. Data Augmentation & Preprocessing Config
# ===========================================================
input_size = 224

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        
        # Augmentation (Train only)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        
        transforms.ToTensor(),
    ]),

    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# ===========================================================
# 3. Custom Dataset Class (Histogram Feature Extraction Added)
# ===========================================================
class MagnesiumDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.all_files = [
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(valid_extensions)
        ]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # --- 读取图像 ---
        try:
            raw_data = np.fromfile(img_path, dtype=np.uint8)
            image_bgr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Read failed: {img_path}, Error: {e}")
            image_bgr = None

        if image_bgr is None:
            return torch.zeros((3, 224, 224)), torch.zeros(64), torch.tensor(0.0)

        # =======================================================
        # 1. 【图像去雾与去模糊】 (前处理)
        # =======================================================
        # 【去雾】通过 YUV 空间直方图均衡化增强对比度
        img_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        temp_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # 【去模糊】使用锐化卷积核增强边缘
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        temp_bgr = cv2.filter2D(temp_bgr, -1, sharpen_kernel)

        # =======================================================
        # 2. 【消除反光与刻痕】 (空间处理)
        # =======================================================
        # 转换到 Lab 空间处理亮度
        image_lab = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(image_lab)

        # 【消除反光/刻痕】识别极高亮区域并进行图像修复 (Inpainting)
        l_median = np.median(l)
        _, mask = cv2.threshold(l, int(l_median + 70), 255, cv2.THRESH_BINARY)
        l = cv2.inpaint(l, mask, 3, cv2.INPAINT_TELEA)

        # =======================================================
        # 3. 【亮度对齐与降噪】 (优化原有逻辑)
        # =======================================================
        l_median_new = np.median(l) 
        shift = 128.0 - l_median_new
        l_aligned = np.clip(l.astype(np.float32) + shift, 0, 255).astype(np.uint8)

        # 【降噪】双边滤波：滤除噪点同时保护边缘
        a_blur = cv2.bilateralFilter(a, 9, 75, 75)
        b_blur = cv2.bilateralFilter(b, 9, 75, 75)
        
        # 这里的变量名改回了你原来的 image_lab_processed
        image_lab_processed = cv2.merge((l_aligned, a_blur, b_blur))
        
        # =======================================================
        # 4. 【直方图与标签】 (完全保留你原来的逻辑)
        # =======================================================
        hist_a = cv2.calcHist([a_blur], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([b_blur], [0], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_feat = np.concatenate([hist_a, hist_b]).flatten()
        hist_feat = torch.tensor(hist_feat, dtype=torch.float32)

        # 保持你原来的 split('_')[1] 解析逻辑
        try:
            temp_str = img_name.split('_')[1]
            temperature = float(temp_str)
        except:
            temperature = 250.0 
            
        label = (temperature - 250.0) / 200.0 
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image_lab_processed) 
            
        return image, hist_feat, label
# ===========================================================
# 4. Dataset Splitting & Loading
# ===========================================================
# 🔴 Please verify your data path
data_dir = r"D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\zhaodu25"  # Update this path accordingly

full_train_ds = MagnesiumDataset(data_dir, transform=data_transforms['train'])
full_test_ds  = MagnesiumDataset(data_dir, transform=data_transforms['test'])

dataset_size = len(full_train_ds)
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))

np.random.shuffle(indices)

test_indices, train_indices = indices[:split], indices[split:]

train_dataset = Subset(full_train_ds, train_indices)
test_dataset  = Subset(full_test_ds, test_indices)

print(f"Data Ready | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ===========================================================
# 5. Model Definition (SE + Histogram Input)
# ===========================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class HybridResNet(nn.Module):
    def __init__(self):
        super(HybridResNet, self).__init__()
        
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.se_block = SEBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 🔥 [Option 1 Update] Statistical Feature Layer
        # Input dim = 8 (Basic Stats) + 64 (Histogram) = 72
        self.stats_fc = nn.Sequential(
            nn.Linear(72, 64), # Expanded neurons
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        # Regression Head
        self.final_regressor = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, hist_vec):
        # --- CNN Branch ---
        feat_map = self.features(x)
        feat_map = self.se_block(feat_map)
        cnn_feat = self.avgpool(feat_map)
        cnn_feat = torch.flatten(cnn_feat, 1)
        
        # --- Statistical Branch ---
        # 1. On-the-fly Basic Stats (8 dims)
        mean_stats = torch.mean(x, dim=[2, 3])
        std_stats = torch.std(x, dim=[2, 3])
        
        mean_a = mean_stats[:, 1:2] 
        mean_b = mean_stats[:, 2:3]
        diff_ab = mean_a - mean_b
        sum_ab  = mean_a + mean_b
        
        basic_stats = torch.cat([mean_stats, std_stats, diff_ab, sum_ab], dim=1)
        
        # 2. 🔥 Concatenate External Histogram Features (64 dims)
        total_stats = torch.cat([basic_stats, hist_vec], dim=1)
        
        stats_out = self.stats_fc(total_stats)
        
        # --- Fusion ---
        combined = torch.cat([cnn_feat, stats_out], dim=1)
        out = self.final_regressor(combined)
        return out

# ===========================================================
# 6. Training Preparation
# ===========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = HybridResNet().to(device)

criterion = nn.SmoothL1Loss()
# Slightly higher initial LR for Cosine Annealing
optimizer = optim.Adam(model.parameters(), lr=0.0005) 

num_epochs = 200

# 🔥 [Option 2] Define Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    
    train_loss_history = []
    test_mae_history = []

    print("-" * 60)
    print(f" Start Training ({num_epochs} Epochs)")
    print("-" * 60)

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        
        # 🔥 Updated loop to accept 3 items
        for inputs, hists, labels in train_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs, hists) # Pass histogram features
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Test ---
        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for inputs, hists, labels in test_loader:
                inputs = inputs.to(device)
                hists = hists.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(inputs, hists)
                
                # Inverse Normalization
                preds_real = outputs * 200.0 + 250.0
                targets_real = labels * 200.0 + 250.0
                
                batch_mae = torch.abs(preds_real - targets_real)
                val_mae_sum += torch.sum(batch_mae).item()
        
        epoch_mae = val_mae_sum / len(test_loader.dataset)
        
        # 🔥 [Option 2] Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record & Print
        train_loss_history.append(epoch_loss)
        test_mae_history.append(epoch_mae)
        
        print(f'Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.6f} | Test MAE: {epoch_mae:.2f}℃')

        # Save Best
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f" New Best! MAE: {best_mae:.2f}℃")

    time_elapsed = time.time() - since
    print("-" * 60)
    print(f'Training Complete. Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f' Final Best Test MAE: {best_mae:.2f}℃')
    
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, test_mae_history

# Execute Training
trained_model, train_hist, test_hist = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)

# Save Model
save_path = 'magnesium_hybrid_hist_model.pth'
torch.save(trained_model.state_dict(), save_path)
print(f"💾 Model saved to: {save_path}")

# ===========================================================
# 7. Visualization
# ===========================================================
def plot_history(train_loss, test_mae):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_mae, 'r-', label='Test MAE')
    plt.title('Test Mean Absolute Error (°C)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_history(train_hist, test_hist)