#include "esp_camera.h"
#include <WiFi.h>
//与其说这里是选择摄像头模式，不如说是根据开发板的不同引脚不同选择模式
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"
#include <WiFiClient.h>
// WiFi的SSID和密码
const char* ssid = "TP-LINK_6BE3";     // 输入你的WiFi名称
const char* passwd = "29826519.com";   // 输入你的WiFi密码
const char* serverIP = "192.168.2.103"; // 目标服务器IP地址
const uint16_t serverPort = 8080; // 目标服务器端口
WiFiClient client;

void setup() {
  // 启动串口通信
  Serial.begin(9600);

  // WiFi模块初始化
  Serial.println("conneting WiFi...");
  
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

  //配置摄像头
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
  config.frame_size = FRAMESIZE_UXGA;
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
  // 循环中的代码
  camera_fb_t *fb = esp_camera_fb_get();
  if(!fb){
    Serial.println("capture image error");
    return ;
  }
  Serial.print("Capture frame size：");
  Serial.println(fb->len);
  // 尝试连接到服务器
  if (client.connect(serverIP, serverPort)) {
    // 发送图像数据
    client.write(fb->buf, fb->len);
    Serial.println("Image data sent");

    client.stop(); // 关闭连接
  } else {
    Serial.println("Connection failed");
  }

  esp_camera_fb_return(fb);
  delay(1000);  // 每隔1秒拍照一次
}
