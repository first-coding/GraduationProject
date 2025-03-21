import os
import torch

# 数据集路径
BASE_DIR = './Facenets/data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')

# 模型参数
IMAGE_SIZE = 128  # 图片的输入尺寸，改为160x160，以适应更复杂的网络结构
BATCH_SIZE = 64  # 每个batch的大小，适中大小有助于较大模型的训练
EPOCHS = 20  # 训练的epochs数，增加训练轮次以保证复杂模型的收敛

# 学习率
LEARNING_RATE = 0.001  # 学习率稍微降低，以适应更深的网络，避免过快下降

# 模型保存路径
MODEL_SAVE_PATH = './Facenets/models/'
MODEL_NAME = 'mobilefacenet_complex.pth'  # 更改模型名称，标记为更复杂的模型

# 是否使用GPU
USE_GPU = True if torch.cuda.is_available() else False  # 根据是否有GPU来选择

# 优化器配置
OPTIMIZER = 'adam'  # 使用Adam优化器，更适合复杂模型
WEIGHT_DECAY = 1e-5  # 使用权重衰减，减少过拟合的风险
MOMENTUM = 0.9  # 动量，用于优化器的加速收敛

# 额外的设置
USE_DATA_AUGMENTATION = True  # 启用数据增强，提高泛化能力
