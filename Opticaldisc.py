import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # 新增进度条库

# ==================== 配置参数 ====================
DATA_DIR = "./Train"  # 包含images和masks子目录的路径
TARGET_SIZE = 512     # 目标尺寸
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_preprocess():
    return A.Compose([
        A.Resize(TARGET_SIZE, TARGET_SIZE),  # 统一调整为512x512
        A.Normalize(mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], is_check_shapes=False)

# ==================== 数据准备 ====================
class RetinaDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.preprocess = get_preprocess()

        # 预验证数据质量
        self._validate_dataset()

    def _validate_dataset(self):
        """执行严格的数据验证"""
        print("\n🔍 正在验证数据集...")
        for idx in tqdm(range(len(self)), desc="数据验证进度"):
            img_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]

            # 验证文件存在性
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件缺失: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"掩膜文件缺失: {mask_path}")

        print("✅ 数据集验证通过")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取并预处理图像
        image = cv2.cvtColor(cv2.imread(self.image_paths[idx]), cv2.COLOR_BGR2RGB)

        # 读取并预处理掩膜（保持二值特性）
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # 直接生成0/1矩阵

        # 应用预处理
        processed = self.preprocess(image=image, mask=mask)
        img_tensor = processed["image"]
        mask_tensor = processed["mask"].float()

        # 最终尺寸验证
        assert img_tensor.shape[1:] == (TARGET_SIZE, TARGET_SIZE), "图像尺寸错误"
        assert mask_tensor.shape == (TARGET_SIZE, TARGET_SIZE), "掩膜尺寸错误"

        return img_tensor, mask_tensor

# ==================== 模型定义 ====================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # 编码器
        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = double_conv(256, 512)

        # 解码器
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up_conv3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv1 = double_conv(128, 64)

        # 输出层
        self.out = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码
        c1 = self.down1(x)  # [B,64,H,W]
        p1 = self.pool(c1)
        c2 = self.down2(p1)  # [B,128,H/2,W/2]
        p2 = self.pool(c2)
        c3 = self.down3(p2)  # [B,256,H/4,W/4]
        p3 = self.pool(c3)

        # 瓶颈
        bn = self.bottleneck(p3)  # [B,512,H/8,W/8]

        # 解码
        u3 = self.up3(bn)  # [B,256,H/4,W/4]
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.up_conv3(u3)

        u2 = self.up2(u3)  # [B,128,H/2,W/2]
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.up_conv2(u2)

        u1 = self.up1(u2)  # [B,64,H,W]
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.up_conv1(u1)

        return self.sigmoid(self.out(u1))


# ==================== 训练函数 ====================
def train():
    # 数据路径处理
    image_dir = os.path.join(DATA_DIR, "images")
    mask_dir = os.path.join(DATA_DIR, "masks")

    # 获取排序后的文件列表
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")))  
    mask_files = sorted(glob(os.path.join(mask_dir, "*.bmp")))  

    # 检查图像和掩膜数量是否匹配
    if len(image_files) != len(mask_files):
        raise ValueError(f"图像数量和掩膜数量不匹配: {len(image_files)} vs {len(mask_files)}")

    # 创建数据集
    train_ds = RetinaDataset(image_files[:int(0.8 * len(image_files))],
                             mask_files[:int(0.8 * len(mask_files))])
    val_ds = RetinaDataset(image_files[int(0.8 * len(image_files)):],
                           mask_files[int(0.8 * len(mask_files)):])

    # 数据加载器
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=4, pin_memory=True)

    # 模型初始化
    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    # 训练循环
    best_dice = 0.0
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
        for images, masks in train_bar:
            images, masks = images.to(DEVICE), masks.unsqueeze(1).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_loss, dice_score = 0.0, 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
            for images, masks in val_bar:
                images, masks = images.to(DEVICE), masks.unsqueeze(1).to(DEVICE)

                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

                preds = (outputs > 0.5).float()
                intersection = (preds * masks).sum()
                union = preds.sum() + masks.sum()
                dice = 2 * intersection / (union + 1e-8)
                dice_score += dice.item()

                val_bar.set_postfix(dice=f"{dice.item():.4f}")

        # 模型保存逻辑
        avg_dice = dice_score / len(val_loader)
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"\n🔥 发现新最佳模型 Dice系数: {best_dice:.4f}")

# ==================== 预测函数 ====================
def predict(image_path, model_path="best_model.pth"):
    # 加载模型
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 读取并预处理
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    preprocess = get_preprocess()
    processed = preprocess(image=image)
    input_tensor = processed["image"].unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # 还原到原始尺寸
    final_mask = cv2.resize(mask, (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST) * 255
    return final_mask


if __name__ == "__main__":

    train()

    test_image = "./Train/images/H0001.jpg"  # 替换为实际路径
    result_mask = predict(test_image)

    cv2.imwrite("predicted_mask.png", result_mask)
    print("Prediction saved to predicted_mask.png")
