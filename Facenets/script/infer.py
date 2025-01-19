import torch
from torchvision import transforms
from PIL import Image
from ..model.MobileFaceNet import MobileFaceNet
from .. import config
import os

# 固定模型路径和类别数
MODEL_PATH = './Facenets/models/mobilefacenet_epoch10.pth'  # 修改为你的模型路径
BASE_DIR = config.BASE_DIR  # 数据集根目录路径

# 载入训练好的模型
def load_model():
    # 获取类别名称，只取文件夹名称（'yes' 和 'no'）
    class_names = [folder for folder in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, folder))]
    class_names = sorted(class_names)  # 确保类别名称排序，'no' 先，'yes' 后

    # 创建模型，类别数根据数据集目录中的文件夹数确定
    model = MobileFaceNet(num_classes=len(class_names))  # 类别数为2
    # 加载模型权重，并允许忽略不匹配的层
    state_dict = torch.load(MODEL_PATH)
    state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(state_dict, strict=False)  # 加载权重时忽略不匹配的层
    model.eval()  # 设置为评估模式
    return model, class_names

# 图像预处理函数
def preprocess_image(image):
    # 定义预处理步骤
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 输入调整为128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 常用的均值和标准差
    ])

    # 将PIL图像应用预处理
    image = transform(image).unsqueeze(0)  # 添加一个batch维度
    return image

# 推断函数，输入PIL图像，返回预测标签和类别名称
def infer_image(image, model, class_names, device):
    # 处理图片并进行推断
    image = preprocess_image(image)
    image = image.to(device)

    with torch.no_grad():  # 不需要计算梯度
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # 获取预测的标签

    predicted_label = predicted.item()  # 预测的标签
    predicted_class_name = class_names[predicted_label]  # 对应的类别名称

    return predicted_label, predicted_class_name  # 返回预测标签和类别名称

# 推断的主函数
def predict(image):
    # 设置设备
    device = torch.device("cuda" if config.USE_GPU else "cpu")

    # 显示输入图片
    # image.show()  # 通过PIL的show()方法显示图片

    # 加载模型和类别名称
    model, class_names = load_model()
    model.to(device)

    # 获取推断结果
    predicted_label, predicted_class_name = infer_image(image, model, class_names, device)

    return predicted_class_name
