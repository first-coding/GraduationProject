from PIL import Image
import os

# 设置输入文件夹路径和输出文件夹路径
input_folder = './Facenets/data/yes'  # 替换为你的输入文件夹路径
output_folder = './Facenets/data/yes'  # 替换为你的输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    # 检查文件是否为图片
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # 打开图片
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # 调整图像大小为128x128
        img_resized = img.resize((128, 128))

        # 保存调整后的图像到输出文件夹
        output_path = os.path.join(output_folder, filename)
        img_resized.save(output_path)

        print(f"Processed: {filename}")

print("All images processed successfully!")
