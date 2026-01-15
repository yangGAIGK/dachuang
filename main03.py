import os
import random
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR 

# ===========================================================
# 1. 基础设置
# ===========================================================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # 为了复现性关闭 benchmark
    print(f"✅ Random seed fixed to: {seed}")

# ===========================================================
# 2. 增强版 Dataset (含 Inpainting 和 坏例过采样)
# ===========================================================
class MagnesiumDataset(Dataset):
    def __init__(self, img_dir, transform=None, is_train=False):
        self.img_dir = img_dir
        self.transform = transform
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        self.all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
        
        # 🟢 [策略三：坏例过采样] 
        # 如果是训练模式，我们把那些已知的"困难样本"复制几份，让模型多学几次
        if is_train:
            # 这里填入你发现的误差大的文件名关键词 'G3_395', 'G4_435', 'G3_385', 'G1_365','G5_400','G3_405','G3_405','G4_350','G3_375','G3_400','G1_340','G3_435','G1_365','G5_450','G5_440','G5_415','G5_390','G5_325'
            hard_samples = ['G13_275','G13_450','G14_375','G13_320','G11_355','G15_250','G15_295','G15_395','G13_315'] 
            extra_files = []
            for f in self.all_files:
                for keyword in hard_samples:
                    if keyword in f:
                        # 对于困难样本，额外复制 3 份放入列表
                        extra_files.extend([f] * 3) 
                        break
            self.all_files.extend(extra_files)
            print(f"🔥 [过采样] 已额外增加 {len(extra_files)} 个困难样本用于训练")

        self.group_map = {
            'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4,
            'G6': 5, 'G7': 6, 'G8': 7, 'G9': 8, 'G10': 9
        }

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            raw_data = np.fromfile(img_path, dtype=np.uint8)
            image_bgr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        except:
            return torch.zeros((3, 224, 224)), torch.zeros(64), torch.tensor(0, dtype=torch.long), torch.tensor(0.0), img_name

        if image_bgr is None:
            return torch.zeros((3, 224, 224)), torch.zeros(64), torch.tensor(0, dtype=torch.long), torch.tensor(0.0), img_name

        # -------------------------------------------------------
        # 🟢 [策略一：自动修补白点] (Highlight Inpainting)
        # -------------------------------------------------------
        # 针对 G2_380_4 这种有亮斑的图
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # 亮度 > 240 的像素被认为是反光点
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        # 只有当反光点存在且面积不大时才修补（避免把整张亮图都修了）
        if cv2.countNonZero(mask) > 0 and cv2.countNonZero(mask) < (image_bgr.shape[0]*image_bgr.shape[1]*0.1):
            image_bgr = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)

        # --- 常规预处理 ---
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        l, a, b = cv2.split(image_lab)

        l_median_new = np.median(l) 
        shift = 128.0 - l_median_new
        l_aligned = l.astype(np.float32) + shift
        l_aligned = np.clip(l_aligned, 0, 255).astype(np.uint8)

        a_blur = cv2.GaussianBlur(a, (5, 5), 0)
        b_blur = cv2.GaussianBlur(b, (5, 5), 0)
        
        image_lab_processed = cv2.merge((l_aligned, a_blur, b_blur))
        
        # --- 直方图 ---
        hist_a = cv2.calcHist([a_blur], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([b_blur], [0], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hist_feat = np.concatenate([hist_a, hist_b]).flatten()
        hist_feat = torch.tensor(hist_feat, dtype=torch.float32)

        # --- 组别解析 ---
        try:
            group_str = img_name.split('_')[0]
            group_id = self.group_map.get(group_str, 0)
        except:
            group_id = 0
        group_id_tensor = torch.tensor(group_id, dtype=torch.long)

        # --- 标签 ---
        try:
            temp_str = img_name.split('_')[1]
            temperature = float(temp_str)
        except:
            temperature = 250.0 
        label = (temperature - 250.0) / 200.0 
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image_lab_processed) 
            
        return image, hist_feat, group_id_tensor, label, img_name

# ===========================================================
# 3. 引入 CBAM 注意力机制 (替代 SEBlock)
# ===========================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 核心：在通道维度上压缩，只看空间信息
        # 这能帮助模型找到"哪里是背景，哪里是划痕"
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    🟢 [策略二：空间注意力]
    相比 SE 只关注通道，CBAM 能关注空间位置。
    它能生成一个掩膜，自动降低 G4_435_8 中那条黑色大划痕的权重。
    """
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x) # 空间加权
        return x

class HybridResNet(nn.Module):
    def __init__(self):
        super(HybridResNet, self).__init__()
        
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        # 🔥 替换为 CBAM
        self.cbam = CBAM(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Group Embedding
        self.group_embed = nn.Embedding(num_embeddings=10, embedding_dim=16)
        
        # Stats FC
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

    def forward(self, x, hist_vec, group_id):
        # CNN + CBAM
        feat_map = self.features(x)
        feat_map = self.cbam(feat_map) # 应用空间注意力
        cnn_feat = self.avgpool(feat_map)
        cnn_feat = torch.flatten(cnn_feat, 1)
        
        # Stats
        mean_stats = torch.mean(x, dim=[2, 3])
        std_stats = torch.std(x, dim=[2, 3])
        mean_a = mean_stats[:, 1:2] 
        mean_b = mean_stats[:, 2:3]
        diff_ab = mean_a - mean_b
        sum_ab  = mean_a + mean_b
        basic_stats = torch.cat([mean_stats, std_stats, diff_ab, sum_ab], dim=1)
        
        # Group
        group_feat = self.group_embed(group_id)
        
        # Concat Stats
        total_stats = torch.cat([basic_stats, hist_vec, group_feat], dim=1)
        stats_out = self.stats_fc(total_stats)
        
        # Weighted Fusion
        weighted_cnn = cnn_feat * torch.abs(self.w_cnn)
        weighted_stats = stats_out * torch.abs(self.w_stats)
        combined = torch.cat([weighted_cnn, weighted_stats], dim=1)
        
        out = self.final_regressor(combined)
        return out

# ===========================================================
# 4. 训练与测试流程
# ===========================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=20):
    device = next(model.parameters()).device
    best_mae = float('inf')
    
    # 历史记录
    train_loss_hist = []
    test_mae_hist = []

    print("-" * 60)
    print(f" Start Training ({num_epochs} Epochs)")
    print("-" * 60)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, hists, groups, labels, _ in train_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            groups = groups.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs, hists, groups)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Test with TTA
        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for inputs, hists, groups, labels, _ in test_loader:
                inputs = inputs.to(device)
                hists = hists.to(device)
                groups = groups.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                # TTA x4
                out1 = model(inputs, hists, groups)
                out2 = model(torch.flip(inputs, [3]), hists, groups)
                out3 = model(torch.flip(inputs, [2]), hists, groups)
                out4 = model(torch.flip(inputs, [2, 3]), hists, groups)
                outputs = (out1 + out2 + out3 + out4) / 4.0
                
                preds_real = outputs * 200.0 + 250.0
                targets_real = labels * 200.0 + 250.0
                val_mae_sum += torch.abs(preds_real - targets_real).sum().item()
        
        epoch_mae = val_mae_sum / len(test_loader.dataset)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss_hist.append(epoch_loss)
        test_mae_hist.append(epoch_mae)
        
        print(f'Epoch {epoch+1:02d}/{num_epochs} | LR: {current_lr:.6f} | Loss: {epoch_loss:.6f} | MAE: {epoch_mae:.2f}℃')

        if epoch_mae < best_mae:
            best_mae = epoch_mae
            torch.save(model.state_dict(), 'best_magnesium_model.pth')
            print(f" ✨ New Best! MAE: {best_mae:.2f}℃")

    return train_loss_hist, test_mae_hist

def analyze_and_plot(model, test_loader, device, error_threshold=10.0):
    model.eval()
    all_preds = []
    all_targets = []
    bad_cases = []
    
    print(f"\n🔍 正在分析坏例 (阈值 > {error_threshold}℃)...")
    
    with torch.no_grad():
        for inputs, hists, groups, labels, filenames in test_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            groups = groups.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # TTA
            out1 = model(inputs, hists, groups)
            out2 = model(torch.flip(inputs, [3]), hists, groups)
            out3 = model(torch.flip(inputs, [2]), hists, groups)
            out4 = model(torch.flip(inputs, [2, 3]), hists, groups)
            outputs = (out1 + out2 + out3 + out4) / 4.0
            
            preds_real = outputs * 200.0 + 250.0
            targets_real = labels * 200.0 + 250.0
            batch_errors = torch.abs(preds_real - targets_real)
            
            all_preds.extend(preds_real.cpu().numpy().flatten())
            all_targets.extend(targets_real.cpu().numpy().flatten())
            
            for i in range(len(filenames)):
                err = batch_errors[i].item()
                if err > error_threshold:
                    bad_cases.append({
                        'name': filenames[i],
                        'actual': targets_real[i].item(),
                        'pred': preds_real[i].item(),
                        'error': err
                    })

    bad_cases.sort(key=lambda x: x['error'], reverse=True)
    print(f"\n🛑 Top 10 Worst Cases:")
    for case in bad_cases[:10]:
        print(f"{case['name']:<15} | Act: {case['actual']:.1f} | Pred: {case['pred']:.1f} | Err: {case['error']:.1f}")

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets, all_preds, color='blue', alpha=0.6)
    min_val = min(min(all_targets), min(all_preds)) - 5
    max_val = max(max(all_targets), max(all_preds)) + 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Temperature'); plt.ylabel('Predicted Temperature')
    plt.title('Prediction vs Actual')
    plt.grid(True, alpha=0.5)
    plt.show()

# ===========================================================
# 5. 主程序
# ===========================================================
if __name__ == '__main__':
    setup_seed(42)
    
    data_dir = r'D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\zhaodu31-35'

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

    # 🔥 开启 is_train=True 触发过采样
    full_train_ds = MagnesiumDataset(data_dir, transform=data_transforms['train'], is_train=True)
    full_test_ds  = MagnesiumDataset(data_dir, transform=data_transforms['test'], is_train=False)

    if len(full_train_ds) > 0:
        dataset_size = len(full_train_ds) # 注意：这里包含了复制出来的坏例
        # 我们需要手动划分，或者使用固定的 validation set
        # 为了简单，我们这里依然使用随机划分，但要注意原本只有几百张图，现在变多了
        
        # 简单起见，我们重新扫描原始文件做索引，避免测试集里混入过采样的复制品
        # 更好的做法是：先划分 Train/Test 文件列表，再在 Train 内部做过采样
        # 但为了不大幅改动结构，我们这里假设 full_train_ds 全部用于训练 (过拟合风险小，因为有大量增强)
        # 而 full_test_ds 我们取一部分不重复的做测试
        
        # 重新设计划分逻辑：
        raw_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        random.shuffle(raw_files)
        split = int(len(raw_files) * 0.2)
        test_files = raw_files[:split]
        train_files = raw_files[split:]
        
        # 构造训练集 (只包含 train_files，并对其中的坏例过采样)
        train_ds = MagnesiumDataset(data_dir, transform=data_transforms['train'], is_train=True)
        # 过滤：只保留属于 train_files 的图片 (包括复制品)
        train_ds.all_files = [f for f in train_ds.all_files if f in train_files or any(k in f for k in ['_copy'])] # 简单过滤逻辑
        
        # 构造测试集 (严格只包含 test_files)
        test_ds = MagnesiumDataset(data_dir, transform=data_transforms['test'], is_train=False)
        test_ds.all_files = test_files
        
        print(f"Data Ready | Train: {len(train_ds)} (含过采样) | Test: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HybridResNet().to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        hist_loss, hist_mae = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200)
        
        # 加载最佳模型进行分析
        model.load_state_dict(torch.load('best_magnesium_model.pth'))
        analyze_and_plot(model, test_loader, device)
    else:
        print("Error: No images.")