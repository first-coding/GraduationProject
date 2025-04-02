#include <Keypad.h>
#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>

#include "esp_camera.h"
#include <WiFi.h>
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"
#include <WiFiClient.h>

// 定义行和列的GPIO引脚
const byte ROWS = 4;  // 行数
const byte COLS = 4;  // 列数
byte rowPins[ROWS] = {1, 0, 45, 47};  // 修正为行连接到的GPIO引脚
byte colPins[COLS] = {21, 46, 3, 38};   // 列连接到的GPIO引脚

// 定义键盘的按键映射
char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

// 创建Keypad对象
Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

// 设定正确的密码
String correctPassword = "223A";  // 修改为你需要的密码
String inputPassword = "";        // 用于保存用户的输入

const int outputPin = 14;  // 定义GPIO14作为输出引脚

// RFID相关设置
#define SS_PIN   42  // SDA 引脚连接到 ESP32 的 GPIO42
MFRC522 mfrc522(SS_PIN, -1);  // 创建一个MFRC522实例，没有使用RST引脚

// 设定特定的卡的UID
byte allowedUID1[] = {0x03, 0x91, 0xB3, 0x28};  // 允许的第一个UID
byte allowedUID2[] = {0x41, 0x40, 0xAC, 0x7B};  // 允许的第二个UID

// WiFi的SSID和密码
const char* ssid     = "OPPO A55 5G";
const char* passwd = "20011008";
const char* serverIP = "192.168.214.210"; // 目标服务器IP地址
const uint16_t serverPort = 8080;       // 目标服务器端口
WiFiClient client;

void setup() {
  Serial.begin(115200);  // 初始化串口
  Serial.println("System Ready.");

  pinMode(outputPin, OUTPUT);  // 设置GPIO43为输出模式
  digitalWrite(outputPin, LOW);  // 初始状态为高电平

  // 初始化Keypad
  Serial.println("4x4 Keypad Ready.");

  // 初始化SPI和RFID
  SPI.begin(41, 39, 40); // 初始化SPI接口, 参数分别为SCK, MISO, MOSI
  mfrc522.PCD_Init();    // 初始化MFRC522模块
  Serial.println("RC522 RFID Reader initialized.");

    WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); //关闭STA模式下wifi休眠，提高响应速度
  // WiFi模块初始化
  Serial.println("Connecting WiFi...");
  
  // 连接WiFi
  WiFi.begin(ssid, passwd);
  
  // 等待WiFi连接
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);  // 每隔1秒检查一次连接状态
    Serial.print('.');  // 打印连接等待中的提示符
  }

  // 连接成功
  Serial.println();
  Serial.println("WiFi Successful!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  // 打印ESP32的IP地址

  // 配置摄像头
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QQVGA;
  config.pixel_format = PIXFORMAT_JPEG;  // for streaming
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 15;
  config.fb_count = 2;

  Serial.println("Initializing camera pins...");
  // 摄像头初始化
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  } else {
    Serial.println("Camera initialized successfully!");
  }

}

void loop() {
  // 检测按键输入
  checkKeypad();
  // 检测RFID卡片
  checkRFID();
  //检测摄像头
  videounlock();
}

void videounlock(){
   if (client.connect(serverIP, serverPort)) {
    // 获取摄像头数据
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Capture image error");
      return;
    }

    // 获取图像数据的大小
    size_t image_size = fb->len;

    // 发送图像的大小
    Serial.println("Sending image size...");
    Serial.println(image_size);
    uint32_t network_image_size = htonl(image_size);  // 转换为大端字节序
// 发送图像大小
   client.write((uint8_t*)&network_image_size, sizeof(network_image_size));
    // 发送图像数据
    Serial.println("Sending image data...");
    client.write(fb->buf, fb->len);  // 发送图像数据
    Serial.println("Image data sent");
    while (client.connected() || client.available()){
    if (client.available()) //如果有数据可读取
    {
      String line = client.readStringUntil('\n'); //读取数据到换行符
      Serial.print("读取到数据：");
      Serial.println(line);
      if (line=="success"){
        triggerOutput();
      }
    }
    }
    // 清理资源
    esp_camera_fb_return(fb);
    client.stop();  // 关闭连接
  } else {
    Serial.println("Connection failed");
  }

  delay(1000);  // 每隔1秒拍照一次
}

void checkKeypad() {
  char key = keypad.getKey();

  if (key) {
    Serial.print("Key pressed: ");
    Serial.println(key);

    if (key == '#') {
      // 用户按下#键时进行密码校验
      if (inputPassword == correctPassword) {
        Serial.println("Password correct!");
        triggerOutput();  // 调用函数设置GPIO43为低电平
      } else {
        Serial.println("Password incorrect!");
      }
      inputPassword = "";  // 清空输入的密码
    } else if (key == 'D') {
      // 用户按下D键时清空当前输入
      inputPassword = "";
      Serial.println("Input cleared.");
    } else {
      // 记录输入的按键
      inputPassword += key;
    }

    // 显示当前输入的密码
    Serial.print("Current input: ");
    Serial.println(inputPassword);
  }
}

void checkRFID() {
  // 检查是否有新的IC卡
  if (!mfrc522.PICC_IsNewCardPresent()) {
    return;
  }

  // 检查是否读取到了卡的UID
  if (!mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  // 打印卡的UID
  Serial.print("Card UID: ");
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
    Serial.print(mfrc522.uid.uidByte[i], HEX);
  }
  Serial.println();

  // 检查卡片的UID是否与允许的UID之一匹配
  if (isUIDMatch(mfrc522.uid.uidByte, mfrc522.uid.size)) {
    Serial.println("Authentication success. Allowed card.");
    triggerOutput();  // 调用函数设置GPIO43为低电平
  } else {
    Serial.println("Authentication failed. Unauthorized card.");
  }

  mfrc522.PICC_HaltA();  // 停止读取卡片
  mfrc522.PCD_StopCrypto1();  // 停止加密
}

// 检查卡片的UID是否与允许的UID匹配
bool isUIDMatch(byte *uid, byte uidSize) {
  // 检查是否与第一个允许的UID匹配
  if (uidSize == sizeof(allowedUID1)) {
    bool match = true;
    for (byte i = 0; i < uidSize; i++) {
      if (uid[i] != allowedUID1[i]) {
        match = false;
        break;
      }
    }
    if (match) return true;  // 与allowedUID1匹配
  }

  // 检查是否与第二个允许的UID匹配
  if (uidSize == sizeof(allowedUID2)) {
    bool match = true;
    for (byte i = 0; i < uidSize; i++) {
      if (uid[i] != allowedUID2[i]) {
        match = false;
        break;
      }
    }
    if (match) return true;  // 与allowedUID2匹配
  }

  return false;  // 不匹配
}

// 控制GPIO43的高低电平
void triggerOutput() {
  digitalWrite(outputPin, HIGH);  // 设置GPIO14为高电平
  delay(2000);  // 保持3秒
  digitalWrite(outputPin, LOW);   // 恢复GPIO14为低电平
}
