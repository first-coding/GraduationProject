import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from torch import nn
from model.MobileFaceNet import MobileFaceNet
import config
from tqdm import tqdm  # 导入tqdm

# 自定义数据集
class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 获取所有类别（文件夹名），并排序
        self.class_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_names.sort()  # 排序，以确保类别的顺序固定
        
        # 为每个类别分配索引
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # 遍历每个类别文件夹
        for label_name in self.class_names:
            class_folder = os.path.join(root_dir, label_name)
            if os.path.isdir(class_folder):  # 确保是文件夹
                # 只处理文件夹中的图像文件
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    # 判断文件是否为图像文件
                    if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图像并确保是RGB格式
        
        if self.transform:
            image = self.transform(image)  # 应用预处理

        return image, label

# 数据预处理和加载
def load_data():
    # 数据增强与预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(degrees=30),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色抖动
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # 随机裁剪并缩放到128x128
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),  # 随机仿射变换
        transforms.GaussianBlur(kernel_size=5),  # 高斯模糊
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 加载数据集，使用自定义的Dataset
    dataset = CustomImageFolder(os.path.join(config.BASE_DIR), transform=transform)
    print("Data classes:", dataset.class_names)  # 打印数据集的类别名
    
    # 计算训练集和测试集的大小，假设80%作为训练集，20%作为测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    # 使用random_split划分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader

# 测试过程
def test_model(model, test_loader, device):
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # 不需要计算梯度
    with torch.no_grad():
        # 使用tqdm显示进度条
        with tqdm(test_loader, unit="batch", desc="Testing") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条和实时损失
                tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    print(f"Test Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.2f}%")

# 训练过程
def train_model():
    # 加载数据
    train_loader, test_loader = load_data()

    # 创建模型
    model = MobileFaceNet(num_classes=2)  # 二分类任务，类别数为2（"yes" 和 "no"）

    # 使用GPU
    device = torch.device("cuda" if config.USE_GPU else "cpu")
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY,betas=(config.MOMENTUM, 0.999))

    # 训练过程
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{config.EPOCHS}") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(device), labels.to(device)

                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)
                loss.backward()

                # 更新权重
                optimizer.step()

                # 累计损失和准确度
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新tqdm进度条和实时损失
                tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        # 打印当前Epoch的训练信息
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # 在每个Epoch后保存模型
        model_save_path = os.path.join(config.MODEL_SAVE_PATH, f"mobilefacenet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)

        # 进行测试
        test_model(model, test_loader, device)

if __name__ == '__main__':
    train_model()