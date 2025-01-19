import os
import shutil
from pathlib import Path

def move_images(src_dir, dst_dir):
    # 确保目标目录存在
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    # 遍历源文件夹（lfw文件夹）
    for root, dirs, files in os.walk(src_dir):
        # 遍历当前文件夹中的所有图片文件
        for file_name in files:
            # 只处理图片文件（根据后缀名）
            if file_name.endswith(('.jpg', '.png', '.jpeg')):
                # 计算图片的完整路径
                img_path = os.path.join(root, file_name)
                
                # 目标路径：将所有文件移动到目标目录，并保持文件名
                target_img_path = os.path.join(dst_dir, file_name)

                # 如果目标文件已存在，可以加上其他处理方式，例如重命名
                # 这里我们选择直接移动
                shutil.move(img_path, target_img_path)
                print(f"Moved {img_path} to {target_img_path}")

# 输入源目录和目标目录
source_directory = "F:/BaiduNetdiskDownload/lfw"  # lfw文件夹路径
destination_directory = "C:/Users/16906/Desktop/SCBC/毕设相关/project/Facenets/data/no"

# 调用函数执行剪切操作
move_images(source_directory, destination_directory)


