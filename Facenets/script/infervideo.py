import cv2
import torch
import numpy as np
from model import MobileFaceNet
import config
from torchvision import transforms
from PIL import Image
import os
# 加载模型
def load_model():
    model = MobileFaceNet(num_classes=None)  # 推理时不需要输出类别
    model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_PATH, config.MODEL_NAME)))
    model.eval()
    return model

def infer_video(model):
    cap = cv2.VideoCapture(0)  # 使用摄像头，0为默认摄像头
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换图像为适合模型输入的格式
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0)  # 添加批次维度

        # 推理
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        # 显示推理结果
        cv2.putText(frame, f"Prediction: {predicted.item()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Inference', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = load_model()
    infer_video(model)
