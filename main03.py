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
from torch.optim.lr_scheduler import CosineAnnealingLR 

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
# 2. 自定义数据集类定义 (全局) - 🟢 [加入组别解析]
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
        
        # 🔥 定义组别映射表
        # 将 G1, G2, G5 等映射为数字 ID
        self.group_map = {
            'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4,
            'G6': 5, 'G7': 6, 'G8': 7, 'G9': 8, 'G10': 9
        }

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
            # 返回空数据防止报错 (Image, Hist, Group, Label)
            return torch.zeros((3, 224, 224)), torch.zeros(64), torch.tensor(0, dtype=torch.long), torch.tensor(0.0)
            
        # Convert to Lab
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        
        l, a, b = cv2.split(image_lab)

        # -------------------------------------------------------
        # Image Cleaning
        # -------------------------------------------------------
        l_median_new = np.median(l) 
        shift = 128.0 - l_median_new
        l_aligned = l.astype(np.float32) + shift
        l_aligned = np.clip(l_aligned, 0, 255).astype(np.uint8)

        a_blur = cv2.GaussianBlur(a, (5, 5), 0)
        b_blur = cv2.GaussianBlur(b, (5, 5), 0)
        
        image_lab_processed = cv2.merge((l_aligned, a_blur, b_blur))
        
        # -------------------------------------------------------
        # 🔥 Calculate Color Histograms
        # -------------------------------------------------------
        hist_a = cv2.calcHist([a_blur], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([b_blur], [0], None, [32], [0, 256])
        
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_feat = np.concatenate([hist_a, hist_b]).flatten()
        hist_feat = torch.tensor(hist_feat, dtype=torch.float32)

        # -------------------------------------------------------
        # 🔥 [新增] 解析组别 ID (Group Embedding)
        # -------------------------------------------------------
        try:
            # 假设文件名格式: G1_435_2.jpg -> 提取 "G1"
            group_str = img_name.split('_')[0]
            # 查表得到 ID，默认给 0 (G1)
            group_id = self.group_map.get(group_str, 0)
        except:
            group_id = 0
            
        group_id_tensor = torch.tensor(group_id, dtype=torch.long) # Embedding层需要 long 类型

        # -------------------------------------------------------
        # Parse Label
        # -------------------------------------------------------
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
            
        # 返回 4 项数据
        return image, hist_feat, group_id_tensor, label

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
        
        # 🔥 [新增] 组别嵌入层 (Positional Embedding)
        # 将组 ID (0-9) 映射为 16 维向量
        self.group_embed = nn.Embedding(num_embeddings=10, embedding_dim=16)
        
        # 🔥 [修改] 统计特征层维度
        # Input dim = 8 (Basic Stats) + 64 (Histogram) + 16 (Group Embedding) = 88
        self.stats_fc = nn.Sequential(
            nn.Linear(88, 64), 
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.w_cnn = nn.Parameter(torch.tensor(1.0))
        self.w_stats = nn.Parameter(torch.tensor(1.5))
        
        self.final_regressor = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x, hist_vec, group_id): # 接收 group_id
        # --- CNN Branch ---
        feat_map = self.features(x)
        feat_map = self.se_block(feat_map)
        cnn_feat = self.avgpool(feat_map)
        cnn_feat = torch.flatten(cnn_feat, 1)
        
        # --- Statistical Branch ---
        mean_stats = torch.mean(x, dim=[2, 3])
        std_stats = torch.std(x, dim=[2, 3])
        mean_a = mean_stats[:, 1:2] 
        mean_b = mean_stats[:, 2:3]
        diff_ab = mean_a - mean_b
        sum_ab  = mean_a + mean_b
        
        basic_stats = torch.cat([mean_stats, std_stats, diff_ab, sum_ab], dim=1) # 8 dims
        
        # 🔥 获取组别特征
        group_feat = self.group_embed(group_id) # [Batch, 16]
        
        # 拼接所有统计类特征 (8 + 64 + 16)
        total_stats = torch.cat([basic_stats, hist_vec, group_feat], dim=1)
        
        stats_out = self.stats_fc(total_stats)
        
        # --- Fusion ---
        weighted_cnn = cnn_feat * torch.abs(self.w_cnn)
        weighted_stats = stats_out * torch.abs(self.w_stats)
        combined = torch.cat([weighted_cnn, weighted_stats], dim=1)
        
        out = self.final_regressor(combined)
        return out

# ===========================================================
# 4. 训练函数定义 (全局) - 🟢 [适配 Group Input]
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
        
        # 🔥 接收 4 个数据项: inputs, hists, groups, labels
        for inputs, hists, groups, labels in train_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            groups = groups.to(device) # 发送到 GPU
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            # 传入 groups
            outputs = model(inputs, hists, groups) 
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Test (with TTA) ---
        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for inputs, hists, groups, labels in test_loader:
                inputs = inputs.to(device)
                hists = hists.to(device)
                groups = groups.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                # 🟢 策略二：测试时增强 (TTA)
                # Group ID 不随翻转改变，直接传入即可
                
                # 1. 预测原图
                out1 = model(inputs, hists, groups)
                
                # 2. 预测水平翻转
                out2 = model(torch.flip(inputs, [3]), hists, groups)
                
                # 3. 预测竖直翻转 (可选，这里加上更稳)
                # out3 = model(torch.flip(inputs, [2]), hists, groups)
                # out4 = model(torch.flip(inputs, [2, 3]), hists, groups)
                # outputs = (out1 + out2 + out3 + out4) / 4.0
                
                # 这里保持你习惯的 2x TTA
                outputs = (out1 + out2) / 2.0
                
                preds_real = outputs * 200.0 + 250.0
                targets_real = labels * 200.0 + 250.0
                
                batch_mae = torch.abs(preds_real - targets_real)
                val_mae_sum += torch.sum(batch_mae).item()
        
        epoch_mae = val_mae_sum / len(test_loader.dataset)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss_history.append(epoch_loss)
        test_mae_history.append(epoch_mae)
        
        print(f'Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.6f} | Loss: {epoch_loss:.6f} | MAE: {epoch_mae:.2f}℃')

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
# 5. 可视化函数定义 (全局) - 🟢 [适配 Group Input]
# ===========================================================
def plot_history(train_loss, test_mae):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_mae, 'r-', label='Test MAE')
    plt.title('Test Mean Absolute Error (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    print("正在进行全量测试集预测以绘制散点图...")
    with torch.no_grad():
        for inputs, hists, groups, labels in test_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            groups = groups.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # TTA 推理
            out1 = model(inputs, hists, groups)
            out2 = model(torch.flip(inputs, [3]), hists, groups)
            outputs = (out1 + out2) / 2.0
            
            preds_real = outputs * 200.0 + 250.0
            targets_real = labels * 200.0 + 250.0
            
            all_preds.extend(preds_real.cpu().numpy().flatten())
            all_targets.extend(targets_real.cpu().numpy().flatten())

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, color='blue', alpha=0.6, label='Predictions')
    min_val = min(min(all_targets), min(all_preds)) - 5
    max_val = max(max(all_targets), max(all_preds)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.xlabel('Actual Temperature (°C)')
    plt.ylabel('Predicted Temperature (°C)')
    plt.title('Prediction vs Actual (With Group Embedding)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    print("✅ 散点图绘制完成")

# ===========================================================
# 6. 主执行逻辑
# ===========================================================
if __name__ == '__main__':
    setup_seed(42)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
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

    # 🔴 请确认你的数据路径
    data_dir = r'D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\zhaodu21-25'

    full_train_ds = MagnesiumDataset(data_dir, transform=data_transforms['train'])
    full_test_ds  = MagnesiumDataset(data_dir, transform=data_transforms['test'])

    dataset_size = len(full_train_ds)
    if dataset_size > 0:
        indices = list(range(dataset_size))
        split = int(np.floor(0.2 * dataset_size))
        np.random.shuffle(indices)
        test_indices, train_indices = indices[:split], indices[split:]

        train_dataset = Subset(full_train_ds, train_indices)
        test_dataset  = Subset(full_test_ds, test_indices)

        print(f"Data Ready | Train: {len(train_dataset)} | Test: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        model = HybridResNet().to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005) 
        num_epochs = 200 # 你可以根据需要调整回200
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        trained_model, train_hist, test_hist = train_model(
            model, train_loader, test_loader, 
            criterion, optimizer, scheduler, 
            num_epochs=num_epochs
        )

        save_path = 'magnesium_hybrid_group_model.pth'
        torch.save(trained_model.state_dict(), save_path)
        print(f"💾 Model saved to: {save_path}")

        plot_history(train_hist, test_hist)
        
        # 加载最佳权重并绘图
        model.load_state_dict(torch.load(save_path))
        plot_scatter(model, test_loader, device)
    else:
        print("❌ 错误：未找到图片，请检查路径！")