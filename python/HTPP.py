import socket
import time

server_ip = '0.0.0.0'  # 监听所有网络接口
server_port = 8080      # 服务器监听端口

# 创建一个TCP套接字
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 绑定IP和端口
server_socket.bind((server_ip, server_port))
server_socket.listen(1)

print(f"服务器正在监听 {server_ip}:{server_port} ...")

# 持续监听客户端连接
while True:
    # 接受一个连接
    client_socket, client_address = server_socket.accept()
    print(f"已连接到 {client_address}")

    # 使用当前时间戳生成唯一的文件名
    timestamp = int(time.time())  # 获取当前时间戳
    filename = f"received_image_{timestamp}.jpg"  # 动态生成文件名

    # 接收数据并保存到动态生成的文件
    with open(filename, 'wb') as f:
        while True:
            data = client_socket.recv(1024)  # 接收数据
            if not data:  # 如果没有数据，则退出循环
                break
            f.write(data)
    
    print(f"图像数据已保存为 {filename}")

    # 注意：这里不关闭连接，保持与客户端的连接

# 关闭服务器套接字（通常不会到达，因为是无限循环）
server_socket.close()
