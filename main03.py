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
# 1. 工具函数定义 (全局)
# ===========================================================
def setup_seed(seed):
    """固定随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed fixed to: {seed}")

# ===========================================================
# 2. 自定义数据集类定义 (全局)
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
        
        # --- Read Image ---
        try:
            raw_data = np.fromfile(img_path, dtype=np.uint8)
            image_bgr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Read failed: {img_path}, Error: {e}")
            image_bgr = None

        if image_bgr is None:
            # Return dummy data if failed (image, hist, label)
            return torch.zeros((3, 224, 224)), torch.zeros(64), torch.tensor(0.0)
            
        # Convert to Lab
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        
        l, a, b = cv2.split(image_lab)

        # -------------------------------------------------------
        # Image Cleaning (Highlight Removal & Alignment)
        # -------------------------------------------------------
        # 1. Highlight Removal (注释保留)
        # l_median = np.median(l)
        # threshold = l_median + 70  
        # mask = l > threshold
        # if np.sum(mask) > 0:
        #     l[mask] = l_median.astype(np.uint8)

        # 2. Global Brightness Alignment
        l_median_new = np.median(l) 
        shift = 128.0 - l_median_new
        l_aligned = l.astype(np.float32) + shift
        l_aligned = np.clip(l_aligned, 0, 255).astype(np.uint8)

        # 3. Denoising
        a_blur = cv2.GaussianBlur(a, (5, 5), 0)
        b_blur = cv2.GaussianBlur(b, (5, 5), 0)
        
        image_lab_processed = cv2.merge((l_aligned, a_blur, b_blur))
        
        # -------------------------------------------------------
        # 🔥 [Option 1] Calculate Color Histograms
        # -------------------------------------------------------
        # Calculate histogram for 'a' channel (32 bins)
        hist_a = cv2.calcHist([a_blur], [0], None, [32], [0, 256])
        # Calculate histogram for 'b' channel (32 bins)
        hist_b = cv2.calcHist([b_blur], [0], None, [32], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Flatten and concatenate -> 64-dim vector
        hist_feat = np.concatenate([hist_a, hist_b]).flatten()
        hist_feat = torch.tensor(hist_feat, dtype=torch.float32)
        # -------------------------------------------------------

        # Parse Label
        try:
            temp_str = img_name.split('_')[1]
            temperature = float(temp_str)
        except:
            temperature = 250.0 
            
        label = (temperature - 250.0) / 200.0 
        label = torch.tensor(label, dtype=torch.float32)

        # Apply Transforms
        if self.transform:
            image = self.transform(image_lab_processed) 
            
        # Return 3 items: Image, Histogram, Label
        return image, hist_feat, label

# ===========================================================
# 3. 模型组件类定义 (全局)
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
# 4. 训练函数定义 (全局)
# ===========================================================
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

# ===========================================================
# 5. 可视化函数定义 (全局)
# ===========================================================
def plot_history(train_loss, test_mae):
    """可视化训练损失和测试MAE曲线"""
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

# ===========================================================
# 6. 主执行逻辑 (封装到 main 代码块)
# ===========================================================
if __name__ == '__main__':
    # =======================================================
    # 6.1 固定随机种子
    # =======================================================
    setup_seed(42)

    # =======================================================
    # 6.2 数据增强 & 预处理配置
    # =======================================================
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

    # =======================================================
    # 6.3 数据集创建 & 拆分
    # =======================================================
    # 🔴 请确认你的数据路径
    data_dir = r'D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\zhaodu11-35'

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

    # =======================================================
    # 6.4 训练准备 (设备、模型、优化器、损失函数)
    # =======================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridResNet().to(device)

    criterion = nn.SmoothL1Loss()
    # Slightly higher initial LR for Cosine Annealing
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 

    num_epochs = 200

    # 🔥 [Option 2] Define Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # =======================================================
    # 6.5 启动训练
    # =======================================================
    trained_model, train_hist, test_hist = train_model(
        model, train_loader, test_loader, 
        criterion, optimizer, scheduler, 
        num_epochs=num_epochs
    )

    # =======================================================
    # 6.6 保存模型 & 可视化结果
    # =======================================================
    # 保存模型权重
    save_path = 'magnesium_hybrid_hist_model.pth'
    torch.save(trained_model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")

    # 可视化训练历史
    plot_history(train_hist, test_hist)