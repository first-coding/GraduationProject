import os
import random

def delete_random_images(directory, target_count=2670):
    # 获取目录中的所有文件（假设目录中只有图片文件）
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # 确保我们有足够的图片
    if len(all_files) <= target_count:
        print(f"文件夹中的图片少于或等于 {target_count} 张，不需要删除。")
        return
    
    # 计算要删除的图片数量
    to_delete_count = len(all_files) - target_count
    
    # 随机选择要删除的文件
    files_to_delete = random.sample(all_files, to_delete_count)
    
    # 删除文件
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        try:
            os.remove(file_path)
            print(f"已删除: {file}")
        except Exception as e:
            print(f"删除 {file} 时出错: {e}")

    print(f"删除完成，剩余图片数量：{target_count}")

# 使用方法
if __name__ == "__main__":
    # 设置你的图片目录路径
    image_directory = "C:/Users/16906/Desktop/SCBC/毕设相关/project/Facenets/data/no"
    
    # 删除随机图片，直到剩下2670张
    delete_random_images(image_directory)
