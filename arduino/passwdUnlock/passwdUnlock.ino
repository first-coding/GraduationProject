#include <Crypto.h>        // 加密库
#include <SHA256.h>        // SHA-256 哈希算法库
#include <EEPROM.h>        // EEPROM 库

#define EEPROM_SIZE 512    // EEPROM 的大小，存储哈希值

// 定义 SHA256 的输出字节大小
#define SHA256_SIZE 32    // SHA-256 哈希值的字节大小

// 定义密码存储的起始位置
#define PASSWORD_HASH_ADDRESS 0

void setup() {
  Serial.begin(115200);  // 初始化串口，便于调试

  // 初始化 EEPROM，大小需要根据存储需求设置
  EEPROM.begin(EEPROM_SIZE);

  // 设置一个简单的操作流程：首先设置密码、验证密码和修改密码
  // 设置密码（用户输入密码后设置）
  String password = "my_secure_password";  // 初始密码
  setPassword(password);  // 设置密码

  // 验证密码（用户输入密码时进行验证）
  String inputPassword = "my_secure_password";  // 用户输入的密码
  if (verifyPassword(inputPassword)) {
    Serial.println("Password is correct!");
  } else {
    Serial.println("Incorrect password.");
  }

  // 修改密码（验证旧密码后修改为新密码）
  String newPassword = "new_secure_password";  // 新密码
  if (changePassword("my_secure_password", newPassword)) {
    Serial.println("Password changed successfully!");
  } else {
    Serial.println("Failed to change password.");
  }

  // 再次验证新密码
  if (verifyPassword(newPassword)) {
    Serial.println("New password is correct!");
  } else {
    Serial.println("New password verification failed.");
  }
}

void loop() {
  // 主循环中可以继续进行其他任务
}

// 设置密码的函数
void setPassword(String password) {
  byte hash[SHA256_SIZE];  // 存储哈希值
  SHA256 sha256;  // 创建 SHA-256 对象
  sha256.update(password.c_str(), password.length());  // 对密码进行哈希加密

  // 使用 finalize 方法并传递正确的参数
  sha256.finalize(hash, SHA256_SIZE);  // 完成哈希运算，并将结果填充到 hash 数组中

  // 将哈希值写入 EEPROM 中
  for (int i = 0; i < SHA256_SIZE; i++) {
    EEPROM.write(PASSWORD_HASH_ADDRESS + i, hash[i]);
  }
  EEPROM.commit();  // 提交写入操作
  Serial.println("Password set successfully.");
}

// 验证密码的函数
bool verifyPassword(String inputPassword) {
  byte inputHash[SHA256_SIZE];  // 存储输入密码的哈希值
  SHA256 sha256;  // 创建 SHA-256 对象
  sha256.update(inputPassword.c_str(), inputPassword.length());  // 对输入密码进行哈希

  // 使用 finalize 方法并传递正确的参数
  sha256.finalize(inputHash, SHA256_SIZE);  // 完成哈希运算，并将结果填充到 inputHash 数组中

  // 从 EEPROM 中读取存储的密码哈希值
  byte storedHash[SHA256_SIZE];
  for (int i = 0; i < SHA256_SIZE; i++) {
    storedHash[i] = EEPROM.read(PASSWORD_HASH_ADDRESS + i);
  }

  // 比较输入的哈希值和存储的哈希值
  for (int i = 0; i < SHA256_SIZE; i++) {
    if (inputHash[i] != storedHash[i]) {
      return false;  // 哈希值不匹配，密码错误
    }
  }
  return true;  // 哈希值匹配，密码正确
}

// 修改密码的函数
bool changePassword(String oldPassword, String newPassword) {
  // 首先验证当前密码
  if (!verifyPassword(oldPassword)) {
    Serial.println("Old password is incorrect.");
    return false;  // 如果验证失败，返回失败
  }

  // 设置新密码
  setPassword(newPassword);  // 使用新密码进行设置
  return true;  // 修改成功
}
