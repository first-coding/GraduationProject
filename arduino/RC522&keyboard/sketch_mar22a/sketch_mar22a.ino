#include <Keypad.h>
#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>
#include "esp_camera.h"
#include <WiFi.h>
#define CAMERA_MODEL_ESP32S3_EYE
#include "camera_pins.h"
#include <WiFiClient.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// WiFi配置
const char* ssid     = "OPPO A55 5G";
const char* passwd   = "20011008";
const char* serverIP = "192.168.57.210";
const uint16_t serverPort = 8080;
WiFiClient client;

// 输出引脚
const int outputPin = 14;

// Keypad配置
const byte ROWS = 4;
const byte COLS = 4;
byte rowPins[ROWS] = {1, 0, 45, 47};
byte colPins[COLS] = {21, 46, 3, 38};

char keys[ROWS][COLS] = {
  {'1', '2', '3', 'A'},
  {'4', '5', '6', 'B'},
  {'7', '8', '9', 'C'},
  {'*', '0', '#', 'D'}
};

Keypad keypad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);
String correctPassword = "223A";
String inputPassword = "";

// RFID配置
#define SS_PIN   42
MFRC522 mfrc522(SS_PIN, -1);
byte allowedUID1[] = {0x03, 0x91, 0xB3, 0x28};
byte allowedUID2[] = {0x41, 0x40, 0xAC, 0x7B};

// BLE配置
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
void triggerOutput();  // 函数声明
class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String value = pCharacteristic->getValue().c_str();
    if (value.length() > 0) {
      Serial.print("收到BLE数据：");
      Serial.println(value);
      if (value == "ok") {
        triggerOutput();
      }
    }
  }
};

void setup() {
  Serial.begin(115200);
  Serial.println("System Ready.");

  // PSRAM检测
  if (psramInit()) {
    Serial.println("PSRAM Initialized Successfully!");
  } else {
    Serial.println("PSRAM Initialization Failed!");
  }

  pinMode(outputPin, OUTPUT);
  digitalWrite(outputPin, LOW);

  // 初始化SPI和RFID
  SPI.begin(41, 39, 40);
  mfrc522.PCD_Init();
  Serial.println("RC522 RFID Reader initialized.");

  // WiFi连接
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  Serial.println("Connecting WiFi...");
  WiFi.begin(ssid, passwd);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print('.');
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // 初始化摄像头
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
  config.xclk_freq_hz = 10000000;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  Serial.println("Initializing camera...");
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  } else {
    Serial.println("Camera initialized successfully!");
  }

  // 初始化BLE
  BLEDevice::init("ESP32-S3-BLE");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);
  BLECharacteristic *pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_WRITE
  );
  pCharacteristic->setCallbacks(new MyCallbacks());
  pService->start();
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->start();
  Serial.println("等待手机BLE连接...");
}

void loop() {
  checkKeypad();
  checkRFID();
  videounlock();
}

void videounlock() {
  if (client.connect(serverIP, serverPort)) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Capture image error");
      return;
    }

    size_t image_size = fb->len;
    uint32_t network_image_size = htonl(image_size);
    client.write((uint8_t*)&network_image_size, sizeof(network_image_size));
    client.write(fb->buf, fb->len);
    Serial.println("图像发送完成");

    while (client.connected() || client.available()) {
      if (client.available()) {
        String line = client.readStringUntil('\n');
        Serial.print("读取到数据：");
        Serial.println(line);
        if (line == "success") {
          triggerOutput();
        }
      }
    }
    esp_camera_fb_return(fb);
    client.stop();
  } else {
    Serial.println("连接服务器失败");
  }
  delay(1000);
}

void checkKeypad() {
  char key = keypad.getKey();
  Serial.println(key);  
  if (key) {
    Serial.print("Key pressed: ");
    Serial.println(key);
    if (key == '#') {
      if (inputPassword == correctPassword) {
        Serial.println("Password correct!");
        triggerOutput();
      } else {
        Serial.println("Password incorrect!");
      }
      inputPassword = "";
    } else if (key == 'D') {
      inputPassword = "";
      Serial.println("Input cleared.");
    } else {
      inputPassword += key;
    }
    Serial.print("Current input: ");
    Serial.println(inputPassword);
  }
}

void checkRFID() {
  if (!mfrc522.PICC_IsNewCardPresent()) return;
  if (!mfrc522.PICC_ReadCardSerial()) return;

  Serial.print("Card UID: ");
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
    Serial.print(mfrc522.uid.uidByte[i], HEX);
  }
  Serial.println();

  if (isUIDMatch(mfrc522.uid.uidByte, mfrc522.uid.size)) {
    Serial.println("Authentication success. Allowed card.");
    triggerOutput();
  } else {
    Serial.println("Authentication failed. Unauthorized card.");
  }

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
}

bool isUIDMatch(byte *uid, byte uidSize) {
  if (uidSize == sizeof(allowedUID1)) {
    bool match = true;
    for (byte i = 0; i < uidSize; i++) {
      if (uid[i] != allowedUID1[i]) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }

  if (uidSize == sizeof(allowedUID2)) {
    bool match = true;
    for (byte i = 0; i < uidSize; i++) {
      if (uid[i] != allowedUID2[i]) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }

  return false;
}

void triggerOutput() {
  digitalWrite(outputPin, HIGH);
  delay(2000);
  digitalWrite(outputPin, LOW);
}
