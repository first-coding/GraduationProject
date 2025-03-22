#include <Keypad.h>
#include <Arduino.h>
#include <SPI.h>
#include <MFRC522.h>

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

const int outputPin = 43;  // 定义GPIO43作为输出引脚

// RFID相关设置
#define SS_PIN   42  // SDA 引脚连接到 ESP32 的 GPIO42
MFRC522 mfrc522(SS_PIN, -1);  // 创建一个MFRC522实例，没有使用RST引脚

// 设定特定的卡的UID
byte allowedUID1[] = {0x03, 0x91, 0xB3, 0x28};  // 允许的第一个UID
byte allowedUID2[] = {0x41, 0x40, 0xAC, 0x7B};  // 允许的第二个UID

void setup() {
  Serial.begin(115200);  // 初始化串口
  Serial.println("System Ready.");

  pinMode(outputPin, OUTPUT);  // 设置GPIO43为输出模式
  digitalWrite(outputPin, HIGH);  // 初始状态为高电平

  // 初始化Keypad
  Serial.println("4x4 Keypad Ready.");

  // 初始化SPI和RFID
  SPI.begin(41, 39, 40); // 初始化SPI接口, 参数分别为SCK, MISO, MOSI
  mfrc522.PCD_Init();    // 初始化MFRC522模块
  Serial.println("RC522 RFID Reader initialized.");
}

void loop() {
  // 检测按键输入
  checkKeypad();

  // 检测RFID卡片
  checkRFID();
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
  digitalWrite(outputPin, LOW);  // 设置GPIO43为低电平
  delay(3000);  // 保持3秒
  digitalWrite(outputPin, HIGH);   // 恢复GPIO43为高电平
}
