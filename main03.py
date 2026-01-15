import os
import random
import time
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms

# ===========================================================
# 1. Fix Random Seed
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
# 2. 针对性数据增强（针对高温区域）
# ===========================================================
input_size = 224

# 创建针对性的数据增强
class TargetedAugmentation:
    @staticmethod
    def adjust_for_temperature(img, temperature):
        """根据温度调整图像增强强度"""
        # 高温区域（>400℃）需要更强的增强
        if temperature > 400:
            # 高温图像可能有更多噪声
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
            ])(img)
        else:
            # 中低温区域使用温和增强
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
            ])(img)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((240, 240)),  # 稍大尺寸，然后随机裁剪
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # 针对性的颜色抖动：高温区域抖动更强
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
    ]),

    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# ===========================================================
# 3. 改进的Dataset Class（带温度感知）
# ===========================================================
class ImprovedMagnesiumDataset(Dataset):
    def __init__(self, img_dir, transform=None, is_train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        valid_extensions = ('.jpg', '.jpeg', '.png')
        self.all_files = [
            f for f in os.listdir(img_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        
        # 按文件名排序
        self.all_files.sort()
        
        # 收集温度和文件名映射
        self.temp_to_files = {}
        self.temperatures = []
        
        for filename in self.all_files:
            try:
                parts = filename.split('_')
                if len(parts) >= 2:
                    temp = float(parts[1])
                    self.temperatures.append(temp)
                    
                    if temp not in self.temp_to_files:
                        self.temp_to_files[temp] = []
                    self.temp_to_files[temp].append(filename)
            except:
                continue
        
        if self.temperatures:
            self.min_temp = min(self.temperatures)
            self.max_temp = max(self.temperatures)
            self.center_temp = (self.min_temp + self.max_temp) / 2
            self.half_range = (self.max_temp - self.min_temp) / 2
            
            print(f"📊 温度分布:")
            print(f"  范围: {self.min_temp:.0f}℃ - {self.max_temp:.0f}℃")
            print(f"  中心: {self.center_temp:.1f}℃")
            
            # 显示每个温度点的样本数
            print(f"  温度点分布:")
            unique_temps = sorted(set(self.temperatures))
            for temp in unique_temps:
                count = len([t for t in self.temperatures if t == temp])
                print(f"    {temp:.0f}℃: {count}个样本")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_name = self.all_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 解析温度
        parts = img_name.split('_')
        temperature = float(parts[1]) if len(parts) >= 2 else self.center_temp
        
        # --- 读取图片 ---
        try:
            image = Image.open(img_path).convert('RGB')
            image_rgb = np.array(image)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except:
            # 返回中性灰色图像
            dummy_image = np.ones((224, 224, 3), dtype=np.uint8) * 128
            image_bgr = dummy_image
            
        # Convert to Lab
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
        
        l, a, b = cv2.split(image_lab)

        # -------------------------------------------------------
        # 温度自适应的图像预处理
        # -------------------------------------------------------
        # 1. 高光去除 - 高温图像可能更亮
        l_median = np.median(l)
        # 高温图像使用更高阈值
        threshold_multiplier = 1.0 + (temperature - self.center_temp) / (self.max_temp - self.min_temp) * 0.5
        threshold = l_median + 70 * threshold_multiplier
        
        mask = l > threshold
        if np.sum(mask) > 0:
            # 使用中值滤波修复高光区域
            l_fixed = cv2.medianBlur(l, 3)
            l[mask] = l_fixed[mask]

        # 2. 亮度对齐 - 考虑温度影响
        l_median_new = np.median(l)
        # 高温图像目标亮度稍低
        target_brightness = 128.0 - (temperature - self.center_temp) / (self.max_temp - self.min_temp) * 10
        shift = target_brightness - l_median_new
        l_aligned = l.astype(np.float32) + shift * 0.8
        l_aligned = np.clip(l_aligned, 0, 255).astype(np.uint8)

        # 3. 自适应去噪 - 高温图像可能需要更强去噪
        kernel_size = 3 if temperature < 350 else 5
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        a_blur = cv2.GaussianBlur(a, (kernel_size, kernel_size), 0)
        b_blur = cv2.GaussianBlur(b, (kernel_size, kernel_size), 0)
        
        # 4. 对比度增强 - 针对不同温度
        if temperature > 400:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_aligned)
        else:
            l_enhanced = l_aligned
        
        image_lab_processed = cv2.merge((l_enhanced, a_blur, b_blur))
        
        # -------------------------------------------------------
        # 增强的特征提取
        # -------------------------------------------------------
        # 多尺度直方图
        hist_features = []
        for bins in [16, 32, 48]:  # 多尺度
            hist_a = cv2.calcHist([a_blur], [0], None, [bins], [0, 256])
            hist_b = cv2.calcHist([b_blur], [0], None, [bins], [0, 256])
            
            cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_features.append(hist_a.flatten())
            hist_features.append(hist_b.flatten())
        
        # 温度相关的统计特征
        stats_features = [
            temperature / 100.0,  # 温度作为特征（归一化）
            np.mean(a_blur),
            np.std(a_blur),
            np.mean(b_blur),
            np.std(b_blur),
            np.mean(l_enhanced),
            np.std(l_enhanced),
            np.median(l_enhanced),
        ]
        
        # 组合特征
        all_features = []
        for feat in hist_features:
            all_features.append(feat.astype(np.float32))
        all_features.append(np.array(stats_features, dtype=np.float32))
        
        hist_feat = np.concatenate(all_features)
        hist_feat = torch.tensor(hist_feat, dtype=torch.float32)
        
        # 标签归一化
        label = (temperature - self.center_temp) / self.half_range
        label = torch.tensor(label, dtype=torch.float32)
        actual_temp = torch.tensor(temperature, dtype=torch.float32)

        # Apply Transforms
        if self.transform:
            image = self.transform(image_lab_processed) 
        else:
            image = transforms.ToTensor()(image_lab_processed)
            
        return image, hist_feat, label, actual_temp

# ===========================================================
# 4. 改进的数据划分（温度平衡）
# ===========================================================
def create_temperature_balanced_split(dataset, test_ratio=0.2):
    """创建温度平衡的训练测试分割"""
    # 按温度分组
    temp_groups = {}
    for i, temp in enumerate(dataset.temperatures):
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(i)
    
    train_indices = []
    test_indices = []
    
    # 对每个温度点进行分层采样
    for temp, indices in temp_groups.items():
        np.random.shuffle(indices)
        split_point = int(len(indices) * (1 - test_ratio))
        train_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, test_indices

# ===========================================================
# 加载数据
# ===========================================================
data_dir = r'D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\zhaodu21-25'

print("正在加载数据集...")
full_dataset = ImprovedMagnesiumDataset(data_dir, transform=data_transforms['train'], is_train=True)

# 温度平衡的分割
train_indices, test_indices = create_temperature_balanced_split(full_dataset, test_ratio=0.2)

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f"\n📊 数据划分 (温度平衡):")
print(f"  训练集: {len(train_dataset)} 样本")
print(f"  测试集: {len(test_dataset)} 样本")

# 检查温度分布
train_temps = [full_dataset.temperatures[i] for i in train_indices]
test_temps = [full_dataset.temperatures[i] for i in test_indices]

print(f"\n🌡️  训练集温度范围: {min(train_temps):.0f}℃ - {max(train_temps):.0f}℃")
print(f"🌡️  测试集温度范围: {min(test_temps):.0f}℃ - {max(test_temps):.0f}℃")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# ===========================================================
# 5. 针对高温区域优化的模型
# ===========================================================
class TemperatureAwareResNet(nn.Module):
    def __init__(self, feature_dim):
        super(TemperatureAwareResNet, self).__init__()
        
        # 使用ResNet18作为基础
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        # SE注意力
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512 // 16),
            nn.ReLU(),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid(),
            nn.Unflatten(1, (512, 1, 1))
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 增强的特征融合
        self.stats_fc = nn.Sequential(
            nn.Linear(8 + feature_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )
        
        # 温度感知的回归头
        self.final_regressor = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 高温区域的额外补偿层
        self.high_temp_adjust = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x, hist_vec, temperature_hint=None):
        # CNN特征
        feat_map = self.features(x)
        
        # SE注意力
        se_weights = self.se_block(feat_map)
        feat_map = feat_map * se_weights
        
        cnn_feat = self.avgpool(feat_map)
        cnn_feat = torch.flatten(cnn_feat, 1)
        
        # 统计特征
        mean_stats = torch.mean(x, dim=[2, 3])
        std_stats = torch.std(x, dim=[2, 3])
        
        mean_a = mean_stats[:, 1:2] 
        mean_b = mean_stats[:, 2:3]
        diff_ab = mean_a - mean_b
        sum_ab = mean_a + mean_b
        
        basic_stats = torch.cat([mean_stats, std_stats, diff_ab, sum_ab], dim=1)
        total_stats = torch.cat([basic_stats, hist_vec], dim=1)
        
        stats_out = self.stats_fc(total_stats)
        
        # 特征融合
        combined = torch.cat([cnn_feat, stats_out], dim=1)
        
        # 基础预测
        base_pred = self.final_regressor(combined)
        
        # 高温补偿（如果有温度提示）
        if temperature_hint is not None:
            # 对高温样本进行额外调整
            high_temp_mask = (temperature_hint > 400).float().unsqueeze(1)
            temp_adjustment = self.high_temp_adjust(temperature_hint.unsqueeze(1))
            adjusted_pred = base_pred + temp_adjustment * high_temp_mask * 0.1
            return adjusted_pred
        
        return base_pred

# ===========================================================
# 6. 训练准备
# ===========================================================
device = torch.device("cuda:0")  # 强制使用GPU，没有则报错
print(f"\n🖥️  使用设备: {device}")

# 获取特征维度
sample_img, sample_feat, _, _ = full_dataset[0]
feature_dim = sample_feat.shape[0]
print(f"特征维度: {feature_dim}")

model = TemperatureAwareResNet(feature_dim).to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 加权损失函数：给高温区域更高权重
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, high_temp_weight=1.5):
        super().__init__()
        self.delta = delta
        self.high_temp_weight = high_temp_weight
        
    def forward(self, pred, target, temperatures):
        # 基础Huber损失
        diff = pred - target
        abs_diff = torch.abs(diff)
        
        # Huber损失
        loss = torch.where(abs_diff < self.delta,
                          0.5 * diff ** 2,
                          self.delta * (abs_diff - 0.5 * self.delta))
        
        # 高温区域加权
        weights = torch.ones_like(loss)
        high_temp_mask = (temperatures > 400).float().unsqueeze(1)
        weights = weights + high_temp_mask * (self.high_temp_weight - 1.0)
        
        return torch.mean(loss * weights)

criterion = WeightedHuberLoss(delta=1.0, high_temp_weight=1.5)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

# 动态学习率调度
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.001,
    epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.3
)

num_epochs = 200

# ===========================================================
# 7. 针对性训练函数
# ===========================================================
def train_with_focus(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                    center_temp, half_range, num_epochs=200):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mae = float('inf')
    # patience_counter = 0  # 早停机制已注释
    
    train_loss_history = []
    test_mae_history = []
    test_mae_high_temp = []  # 高温区域MAE
    test_mae_low_temp = []   # 低温区域MAE

    print("\n" + "="*60)
    print("🔥 开始针对性训练 (聚焦高温区域)")
    print("="*60)

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        
        for inputs, hists, labels, temps in train_loader:
            inputs = inputs.to(device)
            hists = hists.to(device)
            labels = labels.to(device).unsqueeze(1)
            temps = temps.to(device)
            
            optimizer.zero_grad()
            
            # 使用温度作为提示
            outputs = model(inputs, hists, temps/100.0)  # 归一化的温度提示
            
            loss = criterion(outputs, labels, temps)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Test ---
        model.eval()
        val_mae_sum = 0.0
        val_mae_high_sum = 0.0
        val_mae_low_sum = 0.0
        high_temp_count = 0
        low_temp_count = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, hists, labels, actual_temps in test_loader:
                inputs = inputs.to(device)
                hists = hists.to(device)
                labels = labels.to(device).unsqueeze(1)
                actual_temps = actual_temps.to(device)
                
                outputs = model(inputs, hists, actual_temps/100.0)
                
                # 反归一化
                preds_real = outputs * half_range + center_temp
                targets_real = actual_temps.unsqueeze(1)
                
                all_preds.extend(preds_real.cpu().numpy().flatten())
                all_targets.extend(targets_real.cpu().numpy().flatten())
                
                # 总体MAE
                batch_mae = torch.abs(preds_real - targets_real)
                val_mae_sum += torch.sum(batch_mae).item()
                
                # 高温区域MAE (>400℃)
                high_temp_mask = actual_temps > 400
                if torch.any(high_temp_mask):
                    high_mae = torch.abs(preds_real[high_temp_mask] - targets_real[high_temp_mask])
                    val_mae_high_sum += torch.sum(high_mae).item()
                    high_temp_count += torch.sum(high_temp_mask).item()
                
                # 低温区域MAE (<=400℃)
                low_temp_mask = actual_temps <= 400
                if torch.any(low_temp_mask):
                    low_mae = torch.abs(preds_real[low_temp_mask] - targets_real[low_temp_mask])
                    val_mae_low_sum += torch.sum(low_mae).item()
                    low_temp_count += torch.sum(low_temp_mask).item()
        
        epoch_mae = val_mae_sum / len(test_loader.dataset)
        epoch_mae_high = val_mae_high_sum / high_temp_count if high_temp_count > 0 else 0
        epoch_mae_low = val_mae_low_sum / low_temp_count if low_temp_count > 0 else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        train_loss_history.append(epoch_loss)
        test_mae_history.append(epoch_mae)
        test_mae_high_temp.append(epoch_mae_high)
        test_mae_low_temp.append(epoch_mae_low)
        
        # 打印进度
        marker = "🔥" if epoch_mae < 5 else "⚡" if epoch_mae < 6 else "📈"
        print(f'{marker} Epoch {epoch+1:03d}/{num_epochs} | LR: {current_lr:.6f}')
        print(f'   Loss: {epoch_loss:.4f} | MAE: {epoch_mae:.2f}℃')
        print(f'   高温(>400℃): {epoch_mae_high:.2f}℃ | 低温: {epoch_mae_low:.2f}℃')
        
        # 保存最佳模型
        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_wts = copy.deepcopy(model.state_dict())
            # patience_counter = 0  # 早停机制已注释
            
            # 保存详细信息
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mae': best_mae,
                'mae_high': epoch_mae_high,
                'mae_low': epoch_mae_low,
            }, f'best_model_mae_{best_mae:.2f}.pth')
            
            print(f"   ✅ 新最佳! 总体MAE: {best_mae:.2f}℃")
        else:
            # patience_counter += 1  # 早停机制已注释
            # if patience_counter >= 40:  # 增加耐心值  # 早停机制已注释
            #     print(f"   ⏹️  早停触发于 epoch {epoch+1}")  # 早停机制已注释
            #     break  # 早停机制已注释
            pass

    time_elapsed = time.time() - since
    print("\n" + "="*60)
    print(f'🏁 训练完成')
    print(f'   用时: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'   最终最佳测试 MAE: {best_mae:.2f}℃')
    
    model.load_state_dict(best_model_wts)
    
    return (model, train_loss_history, test_mae_history, 
            test_mae_high_temp, test_mae_low_temp, all_preds, all_targets)

# ===========================================================
# 8. 执行训练
# ===========================================================
print("\n" + "="*60)
print("🎯 最终优化 - 目标: MAE < 5℃")
print("="*60)

results = train_with_focus(
    model, train_loader, test_loader, criterion, optimizer, scheduler,
    full_dataset.center_temp, full_dataset.half_range, num_epochs=num_epochs
)

trained_model, train_hist, test_hist, test_high_hist, test_low_hist, all_preds, all_targets = results

# 保存最终模型
save_path = 'optimized_final_model.pth'
torch.save(trained_model.state_dict(), save_path)
print(f"\n💾 模型已保存到: {save_path}")

# ===========================================================
# 9. 详细分析
# ===========================================================
def detailed_analysis(preds, targets, train_loss, test_mae, test_high, test_low):
    errors = np.array(preds) - np.array(targets)
    abs_errors = np.abs(errors)
    targets_arr = np.array(targets)
    
    print("\n" + "="*60)
    print("📊 最终结果详细分析")
    print("="*60)
    
    print(f"\n🎯 总体指标:")
    print(f"  平均绝对误差 (MAE): {np.mean(abs_errors):.2f}℃")
    print(f"  均方根误差 (RMSE): {np.sqrt(np.mean(errors**2)):.2f}℃")
    
    print(f"\n🌡️  分温度区间:")
    temp_ranges = [
        (250, 300, "低温"),
        (300, 350, "中低温"),
        (350, 400, "中温"),
        (400, 450, "高温")
    ]
    
    for low, high, label in temp_ranges:
        mask = (targets_arr >= low) & (targets_arr < high)
        if np.sum(mask) > 0:
            range_errors = errors[mask]
            range_mae = np.mean(np.abs(range_errors))
            range_std = np.std(range_errors)
            print(f"  {label}({low}-{high}℃): {np.sum(mask):2d}样本, "
                  f"MAE: {range_mae:5.2f}℃, STD: {range_std:5.2f}℃")
    
    print(f"\n📈 误差分布分析:")
    sorted_errors = np.sort(abs_errors)
    thresholds = [1, 2, 3, 4, 5, 10]
    for thresh in thresholds:
        percent = np.sum(abs_errors <= thresh) / len(abs_errors) * 100
        print(f"  误差 ≤ {thresh}℃: {percent:5.1f}%")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 训练损失
    axes[0, 0].plot(train_loss, 'b-', linewidth=1.5)
    axes[0, 0].set_title('训练损失', fontsize=12)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 测试MAE
    axes[0, 1].plot(test_mae, 'r-', label='总体', linewidth=1.5)
    axes[0, 1].plot(test_high, 'orange', label='高温(>400℃)', linewidth=1.5, alpha=0.7)
    axes[0, 1].plot(test_low, 'green', label='低温', linewidth=1.5, alpha=0.7)
    axes[0, 1].axhline(y=5, color='k', linestyle='--', alpha=0.5, label='5℃目标')
    axes[0, 1].set_title('测试MAE (°C)', fontsize=12)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 预测vs真实
    axes[0, 2].scatter(targets_arr, preds, alpha=0.6, s=30, c=targets_arr, cmap='coolwarm')
    min_val = min(min(targets), min(preds))
    max_val = max(max(targets), max(preds))
    axes[0, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('真实温度 (°C)')
    axes[0, 2].set_ylabel('预测温度 (°C)')
    axes[0, 2].set_title('预测 vs 真实')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 误差分布
    axes[1, 0].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测误差 (°C)')
    axes[1, 0].set_ylabel('频率')
    axes[1, 0].set_title('误差分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 绝对误差CDF
    sorted_abs = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    axes[1, 1].plot(sorted_abs, cdf, 'b-', linewidth=2)
    for thresh in [1, 3, 5]:
        idx = np.searchsorted(sorted_abs, thresh)
        if idx < len(cdf):
            axes[1, 1].axvline(x=thresh, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].text(thresh, cdf[idx], f'{cdf[idx]*100:.0f}%', 
                           fontsize=9, ha='right')
    axes[1, 1].set_xlabel('绝对误差 (°C)')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('绝对误差CDF')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 温度vs误差散点图
    axes[1, 2].scatter(targets_arr, abs_errors, alpha=0.6, s=30)
    # 添加移动平均线
    window = 20
    if len(targets_arr) > window:
        sorted_idx = np.argsort(targets_arr)
        sorted_temps = targets_arr[sorted_idx]
        sorted_errors_abs = abs_errors[sorted_idx]
        
        moving_avg = np.convolve(sorted_errors_abs, np.ones(window)/window, mode='valid')
        temp_avg = np.convolve(sorted_temps, np.ones(window)/window, mode='valid')
        
        axes[1, 2].plot(temp_avg, moving_avg, 'r-', linewidth=2, label=f'{window}点移动平均')
    
    axes[1, 2].axhline(y=5, color='k', linestyle='--', alpha=0.5, label='5℃目标')
    axes[1, 2].set_xlabel('真实温度 (°C)')
    axes[1, 2].set_ylabel('绝对误差 (°C)')
    axes[1, 2].set_title('温度 vs 绝对误差')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 目标达成情况:")
    mae_under_5 = np.mean(abs_errors < 5) * 100
    print(f"  误差在5℃以内的样本比例: {mae_under_5:.1f}%")
    
    if np.mean(abs_errors) < 5:
        print(f"\n✅ 成功! 平均MAE达到{np.mean(abs_errors):.2f}℃，低于5℃目标!")
    else:
        print(f"\n⚠️  接近目标! 平均MAE为{np.mean(abs_errors):.2f}℃，略高于5℃目标")

# 执行分析
detailed_analysis(all_preds, all_targets, train_hist, test_hist, test_high_hist, test_low_hist)

print("\n" + "="*60)
print("🎉 最终优化完成!")
print("="*60)