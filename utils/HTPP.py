import socket
import time
from io import BytesIO
from PIL import Image
from Facenets.script.infer import predict

server_ip = '0.0.0.0'  # 监听所有网络接口
server_port = 8080      # 服务器监听端口

# 创建TCP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP和端口
server_socket.bind((server_ip, server_port))
server_socket.listen(1)

print(f"服务器正在监听 {server_ip}:{server_port} ...")

while True:
    try:
        # 接受一个连接
        client_socket, client_address = server_socket.accept()
        print(f"已连接到 {client_address}")

        # 先接收图像的大小（4字节）
        image_size_data = client_socket.recv(4)
        if not image_size_data:
            print("没有收到图像大小")
            continue

        # 获取图像大小
        expected_image_size = int.from_bytes(image_size_data, 'big')
        print(f"预期接收图像大小: {expected_image_size} 字节")
        # 接收完整的图像数据
        image_data = b""
        total_data_size = 0
        while total_data_size < expected_image_size:
            data = client_socket.recv(2048)
            if not data:
                print(f"客户端 {client_address} 已断开连接")
                break
            image_data += data
            total_data_size += len(data)
            print(f"接收到 {len(data)} 字节，当前数据大小: {total_data_size} 字节")

        if total_data_size == expected_image_size:
            print("完整接收到图像数据")

            # 使用BytesIO将接收到的字节数据转换为PIL图像
            try:
                image = Image.open(BytesIO(image_data)).convert('RGB')
            except Exception as e:
                print(f"图像解码失败: {e}")
                continue

            print("接收到图像，正在进行推断...")

            # 调用predict函数进行推断
            predicted_class_name = predict(image)  # 直接传入PIL图像对象

            # 打印预测结果
            print(f"预测标签名称: {predicted_class_name}")

            # 根据预测标签，发送响应给客户端
            if predicted_class_name == 'yes':
                client_socket.sendall(b"success")
            else:
                client_socket.sendall(b"failure")

            time.sleep(0.1)  # 等待一段时间，确保消息完全发送

        else:
            print(f"接收到的数据不完整，预期接收 {expected_image_size} 字节，实际接收 {total_data_size} 字节")

        # 关闭与客户端的连接
        client_socket.close()

        # 每次接收完图像数据后，重置 `total_data_size` 为 0
        total_data_size = 0

    except Exception as e:
        print(f"服务器错误: {e}")
        continue

# 关闭服务器套接字（通常不会到达，因为是无限循环）
server_socket.close()
