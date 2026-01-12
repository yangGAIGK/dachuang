import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

# ============================================================
# 1. 重新定义 HybridResNet 网络结构
# (必须与 main02.py 中的定义完全一致，否则无法加载权重)
# ============================================================
class HybridResNet(nn.Module):
    def __init__(self):
        super(HybridResNet, self).__init__()
        
        # 加载 ResNet18 骨架 (推理时不需要下载预训练权重，设为 None 即可)
        base_model = models.resnet18(weights=None)
        
        # 提取卷积特征部分 (去掉最后两层)
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 统计特征层
        self.stats_fc = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        # 回归头
        self.final_regressor = nn.Sequential(
            nn.Linear(512 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [Batch, 3, 224, 224]
        
        # CNN 分支
        feat_map = self.features(x)
        cnn_feat = self.avgpool(feat_map)
        cnn_feat = torch.flatten(cnn_feat, 1)
        
        # 统计特征分支 (Mean/Std)
        mean_stats = torch.mean(x, dim=[2, 3])
        std_stats = torch.std(x, dim=[2, 3])
        stats = torch.cat([mean_stats, std_stats], dim=1)
        stats_out = self.stats_fc(stats)
        
        # 融合与预测
        combined = torch.cat([cnn_feat, stats_out], dim=1)
        out = self.final_regressor(combined)
        return out

def get_model(device):
    model = HybridResNet()
    model = model.to(device)
    return model

# ============================================================
# 2. 定义推理函数
# ============================================================
def predict_temperature(image_path, model_path, device):
    # --- A. 加载模型 ---
    print(f"正在加载模型权重: {model_path} ...")
    
    model = get_model(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 模型加载成功！")
    else:
        print(f"❌ 错误：找不到模型文件 {model_path}")
        return None

    model.eval() # 切换到评估模式

    # --- B. 读取与预处理图片 ---
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 {image_path}")
        return None

    # 1. 使用 OpenCV 读取 (解决中文路径问题)
    try:
        raw_data = np.fromfile(image_path, dtype=np.uint8)
        img_bgr = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        if img_bgr is None: raise ValueError("解码失败")
    except Exception as e:
        print(f"读取图片失败: {e}")
        return None

    # 2. 颜色空间转换 (必须转为 Lab，与训练时保持一致！)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # 先转 RGB 供显示用
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab) # 再转 Lab 供模型用

    # 3. 定义 Transform (与训练代码中的 test transform 一致)
    # 注意：训练时去掉了 ImageNet Normalize，只用了 Resize 和 ToTensor
    transform = transforms.Compose([
        transforms.ToPILImage(), # 接受 Lab numpy 数组
        transforms.Resize((224, 224)),
        transforms.ToTensor(),   # 归一化到 [0, 1]
    ])
    
    # 应用变换
    img_tensor = transform(img_lab) 
    img_tensor = img_tensor.unsqueeze(0) # 增加 Batch 维度 [1, 3, 224, 224]
    img_tensor = img_tensor.to(device)

    # --- C. 预测 ---
    with torch.no_grad():
        output = model(img_tensor)
        pred_normalized = output.item()

    # --- D. 反归一化 ---
    # 公式: T = val * 200 + 250
    pred_temp = pred_normalized * 200.0 + 250.0

    return img_rgb, pred_temp

# ============================================================
# 3. 主程序入口
# ============================================================
if __name__ == '__main__':
    # -------------------------------------------------------------
    # 🔴 请确保路径正确
    MODEL_PATH = 'magnesium_resnet01_model.pth'  
    
    # 测试图片路径 (支持中文)
    TEST_IMG_PATH = r"D:\Study\大三上\science\大创\JPG-处理图\JPG-处理图\test\G10_445_10.jpg"
    # -------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    # 执行预测
    result = predict_temperature(TEST_IMG_PATH, MODEL_PATH, device)

    if result:
        img_vis, temp = result # img_vis 是 RGB 格式，方便 matplotlib 显示
        
        print("\n" + "="*30)
        print(f"📄 图片: {os.path.basename(TEST_IMG_PATH)}")
        print(f"🌡️ 预测温度: {temp:.2f} ℃")
        print("="*30 + "\n")

        # 显示图像
        plt.figure(figsize=(6,6))
        plt.imshow(img_vis)
        plt.title(f"Predicted: {temp:.2f} C")
        plt.axis('off')
        plt.show()