#include <SPI.h>
#include <MFRC522.h>

// 没有RST引脚，所以我们忽略 RST 引脚定义
#define SS_PIN   42  // SDA 引脚连接到 ESP32 的 GPIO42

MFRC522 mfrc522(SS_PIN, -1);  // 创建一个MFRC522实例，没有使用RST引脚

void setup() {
  Serial.begin(115200);  // 初始化串口
  SPI.begin(41, 39, 40); // 初始化SPI接口, 参数分别为SCK, MISO, MOSI
  mfrc522.PCD_Init();    // 初始化MFRC522模块
  Serial.println("RC522 RFID Reader initialized.");
}

void loop() {
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

  // 假设要使用默认的密钥
  MFRC522::MIFARE_Key key;
  for (byte i = 0; i < 6; i++) {
    key.keyByte[i] = 0xFF;  // 默认的MIFARE密钥为FF FF FF FF FF FF
  }

  // 验证卡片是否能被读取
  if (mfrc522.PCD_Authenticate(MFRC522::PICC_CMD_MF_AUTH_KEY_A, 8, &key, &(mfrc522.uid)) == MFRC522::STATUS_OK) {
    Serial.println("Authentication success.");
  } else {
    Serial.println("Authentication failed.");
  }

  mfrc522.PICC_HaltA();  // 停止读取卡片
  mfrc522.PCD_StopCrypto1();  // 停止加密
}
