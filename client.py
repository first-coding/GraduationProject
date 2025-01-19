import socket
import time
from PIL import Image
from io import BytesIO

# 创建客户端套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 连接服务器
server_ip = '192.168.2.102'
server_port = 8080
client_socket.connect((server_ip, server_port))

# 打开图像并准备发送
image_path = "image.jpg"  # 你的图像文件路径
image = Image.open(image_path)

# 将图像转为字节流
img_byte_arr = BytesIO()
image.save(img_byte_arr, format='JPEG')
img_data = img_byte_arr.getvalue()

# 发送图像的大小
image_size = len(img_data)
client_socket.sendall(image_size.to_bytes(4, 'big'))  # 发送4字节的整数作为图像大小

# 发送图像数据
client_socket.sendall(img_data)

# 等待服务器的响应
response = client_socket.recv(1024)
print(f"收到来自服务器的响应: {response.decode()}")

# 关闭连接
client_socket.close()
