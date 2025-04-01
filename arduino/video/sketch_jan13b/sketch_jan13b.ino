#include "esp_camera.h"
#include <WiFi.h>
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"
#include <WiFiClient.h>

// WiFi的SSID和密码
const char* ssid = "TP-LINK_6BE3";     // 输入你的WiFi名称
const char* passwd = "29826519.com";   // 输入你的WiFi密码
const char* serverIP = "192.168.2.100"; // 目标服务器IP地址
const uint16_t serverPort = 8080;       // 目标服务器端口
WiFiClient client;

void setup() {
  // 启动串口通信
  Serial.begin(9600);
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
        pinMode(14,OUTPUT);
        digitalWrite(14, HIGH);
        delay(3000);
        digitalWrite(14, LOW);
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